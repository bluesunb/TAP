import os
import numpy as np
from torch.utils import data

from latent_planner.config import DatasetConfig
from latent_planner.datasets.preprocessing import dataset_preprocess_functions
from latent_planner.datasets.d4rl_utils import load_environment, suppress_output, get_dataset
from latent_planner.datasets.d4rl_utils import make_env

from typing import Optional, Tuple, Dict, Any, List


def segment(episodes: np.ndarray,
            terminals: np.ndarray,
            max_path_length: int,
            min_path_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        episodes: (T, D) : all episodes concatenated
        terminals: (T, ) : all terminals concatenated
        max_path_length: := L

    Returns:
        trajectory: (N, L, trans_dim)
        early_terminals: (N, L) : 1 if terminal, 0 otherwise
        ep_lens: (N, ) : length of each trajectory

    """
    assert len(episodes) == len(terminals), \
        "len(observations)={len(observations)} != len(terminals)={len(termianls)}"

    transition_dim = episodes.shape[1]
    raw_trajectories = np.split(episodes, np.where(terminals)[0] + 1, axis=0)
    ep_lens = np.array(list(map(len, raw_trajectories)))
    ep_ids = np.where(ep_lens >= min_path_length)[0]
    n_traj = len(ep_ids)

    assert np.all(ep_lens <= max_path_length), \
        f"max_path_length({max_path_length} < max ep length ({ep_lens.max()})"

    trajectory = np.zeros((n_traj, max_path_length, transition_dim), dtype=np.float32)
    early_terminals = np.zeros((n_traj, max_path_length), dtype=int)
    for i in range(n_traj):
        ep_len = ep_lens[i]
        trajectory[i, :ep_len] = raw_trajectories[ep_ids[i]]
        early_terminals[i, ep_len:] = 1

    return trajectory, early_terminals, ep_lens


def discount_cumsum(x: np.ndarray, discount: float):
    cumsum = np.zeros_like(x)
    cumsum[..., -1] = x[..., -1]
    for i in reversed(range(x.shape[-1] - 1)):
        cumsum[..., i] = x[..., i] + discount * cumsum[..., i + 1]
    return cumsum

class SeqDataset(data.Dataset):
    def __init__(self, config: DatasetConfig):
        print(f'[ datasets/sequence ] Sequence length: {config.seq_len} |'
              f' Step: {config.step} |'
              f' Max path length: {config.max_path_length}')

        self.config = config
        self.seq_len = config.seq_len
        self.step = config.step
        # self.dataset = self.load_dataset()
        # dataset = get_dataset(make_env(config.env_name), config.env_name, filter_terminals=False)
        dataset = get_dataset(config.env_name, filter_terminals=False)

        observations = dataset['observations']
        actions = dataset['actions'].astype(np.float32)
        rewards = dataset['rewards'].astype(np.float32).reshape(-1, 1)
        terminals = dataset['terminals'].reshape(-1, 1)

        self.observation_dim = observations.shape[-1]
        self.action_dim = actions.shape[-1]

        self.obs_mean, self.obs_std = self.get_stats(observations, axis=0, keepdims=True)
        self.act_mean, self.act_std = self.get_stats(actions, axis=0, keepdims=True)
        # self.rew_mean, self.rew_std = self.get_stats(rewards)

        term_mask = terminals.squeeze() == 1
        if config.termination_penalty is not None:
            rewards[term_mask] = config.termination_penalty
            self.rew_mean, self.rew_std = self.get_stats(rewards[~term_mask])
        else:
            self.rew_mean, self.rew_std = self.get_stats(rewards)

        if config.normalize_sa:
            observations = (observations - self.obs_mean) / self.obs_std
            actions = (actions - self.act_mean) / self.act_std

        sar, early_terminals, ep_lens = segment(
            np.concatenate([observations, actions, rewards], axis=-1),
            terminals, config.max_path_length, config.min_path_length)

        self.term_mask = early_terminals
        self.ep_lens = ep_lens
        self.cum_ep_lens = np.cumsum(ep_lens)

        values = discount_cumsum(sar[:, :, -1], config.discount)
        self.value_mean, self.value_std = 0, 1

        if config.normalize_sa and config.normalize_reward:
            self.value_mean, self.value_std = self.get_stats(
                values.flatten()[early_terminals.flatten() != 1])

            # rewards = (sar[:, :, -1] - self.rew_mean) / self.rew_std
            # rewards = np.where(early_terminals, sar[:, :, -1], (sar[:, :, -1] - self.rew_mean) / self.rew_std)
            if 'antmaze' in config.env_name:
                rewards[~term_mask] = rewards[~term_mask] - 1
            else:
                rewards[~term_mask] = (rewards[~term_mask] - self.rew_mean) / self.rew_std
            values = (values - self.value_mean) / self.value_std
        rewards, _, _ = segment(rewards, terminals, config.max_path_length, config.min_path_length)

        self.episodes = np.concatenate([sar[:, :, :-1], rewards, values[:, :, None]], axis=-1)
        self.episodes = np.pad(self.episodes, ((0, 0), (0, config.seq_len - 1), (0, 0)))
        self.term_mask = np.pad(self.term_mask, ((0, 0), (0, config.seq_len - 1)), constant_values=1)

        self.mean = np.concatenate([self.obs_mean[0], self.act_mean[0], [self.rew_mean, self.value_mean]])
        self.std = np.concatenate([self.obs_std[0], self.act_std[0], [self.rew_std, self.value_std]])

    # def load_dataset(self):
    #     env_name = self.config.env_name
    #     dataset_name = self.config.dataset
    #     print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
    #     dataset = get_trajectories(env_name, dataset_name,
    #                                terminate_on_end=True, disable_goal=self.config.disable_goal)
    #     print(f'[ datasets/sequence ] Done')
    #
    #     preprocess_fn = dataset_preprocess_functions.get(env_name)
    #     if preprocess_fn:
    #         print(f'[ datasets/sequence ] Preprocessing...', end=' ', flush=True)
    #         dataset = preprocess_fn(dataset)
    #         print(f'[ datasets/sequence ] Done')
    #
    #     return dataset

    def get_pos(self, idx):
        ep_idx = np.searchsorted(self.cum_ep_lens, idx, side='right')
        delta = idx - self.cum_ep_lens[ep_idx - 1] if ep_idx > 0 else idx
        return ep_idx, delta

    @staticmethod
    def get_stats(x: np.ndarray, axis=None, keepdims=False):
        return x.mean(axis=axis, keepdims=keepdims), x.std(axis=axis, keepdims=keepdims)

    def normalize_joined_single(self, joined):
        return (joined - self.mean) / self.std

    def __len__(self):
        return (np.maximum(0, self.ep_lens - self.seq_len) + 1).sum()

    def __getitem__(self, idx):
        """
        48 micro-sec +- 326 ns
        """
        ep_idx, delta = self.get_pos(idx)
        mask = (np.arange(self.seq_len) + delta) < (self.config.max_path_length - self.step)
        term_mask = self.term_mask[ep_idx, delta:delta + self.seq_len:self.step, None]

        episode = self.episodes[ep_idx, delta:delta + self.seq_len:self.step]
        prev = episode[:-1]
        now = episode[1:]
        mask = np.repeat(mask.reshape(-1, 1), prev.shape[1], axis=1)[:-1]
        term_mask = term_mask.reshape(-1, 1)[:-1]

        return prev, now, mask, term_mask


if __name__ == "__main__":
    from latent_planner.config import DefaultConfig, DatasetConfig
    from latent_planner.datasets import sequence

    config = DefaultConfig()

    # ============= Dataset =============

    env_name = "halfcheetah-medium-expert-v0"
    dataset = None
    # env_name = "maze2d-medium-v1"
    # dataset = 'maze2d-medium-sparse-v1'
    # env = datasets.load_environment(env_name)
    config.env_name = env_name
    config.dataset = dataset

    seq_len = config.subsampled_seq_len * config.step
    config.log_dir = os.path.expanduser(config.log_dir)
    config.save_dir = os.path.expanduser(config.save_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    dataset_class = SeqDataset
    dataset_config = DatasetConfig(
        save_dir=(config.save_dir, 'data_config.pkl'),
        env_name=config.env_name,
        dataset=config.dataset,
        termination_penalty=config.termination_penalty,
        seq_len=seq_len,
        step=config.step,
        discount=config.discount,
        disable_goal=config.disable_goal,
        normalize_sa=config.normalize_state,
        normalize_reward=config.normalize_reward,
        max_path_length=int(config.max_path_length),
        device=config.device
    )

    x = np.random.randn(10000, 4)
    terms = np.random.choice(2, size=10000, p=[0.9, 0.1])
    max_path_length = 1000
    sar, early_terminals, ep_lens = segment(x, terms, max_path_length)
    sar2, early_terminals2, ep_lens2 = sequence.segment(x, terms, max_path_length)

    assert np.allclose(sar, sar2)
    assert np.allclose(early_terminals, early_terminals2)

    dataset = SeqDataset(dataset_config)
    dataset2 = sequence.SequenceDataset(dataset_config)

    now, nxt, mask, term = dataset[1234]
    now2, nxt2, mask2, term2 = dataset2[1234]

    assert np.allclose(now, now2)
    assert np.allclose(nxt, nxt2)
    assert np.allclose(mask, mask2)
    assert np.allclose(term, term2)
