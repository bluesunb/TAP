import os
import numpy as np
from torch.utils import data

from latent_planner.config import DatasetConfig
from latent_planner.datasets.preprocessing import dataset_preprocess_functions
from latent_planner.datasets.d4rl_utils import load_environment, suppress_output, get_dataset, make_env
from latent_planner.datasets.dataset_utils import normalize, denormalize

from typing import Optional, Tuple, Dict, Any, List


def discount_cumsum(x: np.ndarray, discount: float):
    cumsum = np.zeros_like(x)
    cumsum[..., -1] = x[..., -1]
    for i in reversed(range(x.shape[-1] - 1)):
        cumsum[..., i] = x[..., i] + discount * cumsum[..., i + 1]
    return cumsum


class SeqDataset(data.Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.seq_len = config.seq_len
        self.step = config.step

        dataset = get_dataset(config.env_name, filter_terminals=False)
        transition_keys = ['observations', 'actions', 'rewards', 'terminals']
        dims = np.cumsum([dataset[k].shape[-1] for k in transition_keys[:-1]])
        episodes = np.concatenate([dataset[k] for k in transition_keys], axis=-1)
        episodes, after_term, ep_lens = self.segment(episodes, dataset['truncated'].flatten())
        self.observations, self.actions, self.rewards, self.terminals = np.split(episodes, dims, axis=-1)

        self.observation_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]

        self.obs_mean, self.obs_std = self.get_stats(self.observations, axis=(0, 1))
        self.act_mean, self.act_std = self.get_stats(self.actions, axis=(0, 1))
        self.rew_mean, self.rew_std = self.get_stats(self.rewards)
        if config.normalize_sa:
            self.observations = normalize(self.observations, self.obs_mean, self.obs_std)
            self.actions = normalize(self.actions, self.act_mean, self.act_std)

        if config.termination_penalty:
            rewards = dataset['rewards']
            rewards[dataset['truncated'].squeeze() == 1] = config.termination_penalty
            self.rewards = self.segment(rewards, dataset['truncated'].flatten())[0]

        self.values = discount_cumsum(self.rewards.squeeze(), config.discount)
        self.value_mean, self.value_std = 0.0, 1.0
        if config.normalize_reward:
            self.value_mean, self.value_std = self.get_stats(self.values)
            self.values = normalize(self.values, self.value_mean, self.value_std)
            self.rewards = normalize(self.rewards, self.rew_mean, self.rew_std)

        self.ep_term_mask = np.pad(after_term, ((0, 0), (0, config.seq_len - 1)), constant_values=1)
        self.episodes = np.concatenate(
            [self.observations, self.actions, self.rewards, self.values[..., None]], axis=-1)
        self.episodes = np.pad(self.episodes, ((0, 0), (0, config.seq_len - 1), (0, 0)))

        self.cum_ep_lens = np.cumsum(ep_lens)
        self.mean = np.concatenate([self.obs_mean, self.act_mean, [self.rew_mean, self.value_mean]])
        self.std = np.concatenate([self.obs_std, self.act_std, [self.rew_std, self.value_std]])

    def __len__(self):
        return int(self.cum_ep_lens[-1])

    def __getitem__(self, idx):
        ep_idx, delta = self.get_pos(idx)
        mask = (np.arange(self.seq_len) + delta) < (self.config.max_path_length - self.step)
        term_mask = self.ep_term_mask[ep_idx, delta:delta + self.seq_len:self.step, None]

        episode = self.episodes[ep_idx, delta:delta + self.seq_len:self.step]
        current = episode[:-1]
        next_ = episode[1:]
        mask = np.repeat(mask.reshape(-1, 1), current.shape[1], axis=1)[:-1]
        term_mask = term_mask.reshape(-1, 1)[:-1]

        return current, next_, mask.astype(np.float32), term_mask.astype(np.float32)

    def get_pos(self, idx):
        ep_idx = np.searchsorted(self.cum_ep_lens, idx, side='right')
        delta = idx - self.cum_ep_lens[ep_idx - 1] if ep_idx > 0 else idx
        return ep_idx, delta

    @staticmethod
    def get_stats(data: np.ndarray, axis=None, keepdims=False):
        return (np.mean(data, axis=axis, keepdims=keepdims),
                np.std(data, axis=axis, keepdims=keepdims))

    def segment(self, data: np.ndarray, terminals: np.ndarray):
        episodes = np.split(data, np.where(terminals)[0] + 1, axis=0)
        ep_lens = np.array(list(map(len, episodes)))
        ep_ids = np.where(ep_lens >= self.config.min_path_length)[0]

        assert np.all(ep_lens <= self.config.max_path_length), \
            f"max_path_length({self.config.max_path_length} < max ep length ({ep_lens.max()})"

        episodes_arr = np.zeros((len(ep_ids), self.config.max_path_length, data.shape[-1]), dtype=np.float32)
        after_terminals = np.zeros((len(ep_ids), self.config.max_path_length), dtype=np.float32)
        for i in range(len(ep_ids)):
            ep_len = ep_lens[i]
            episodes_arr[i, :ep_len] = episodes[ep_ids[i]]
            after_terminals[i, ep_len:] = 1

        ep_lens = ep_lens[ep_lens >= self.config.min_path_length]
        return episodes_arr, after_terminals, ep_lens


if __name__ == "__main__":
    from latent_planner.config import DefaultConfig, DatasetConfig
    from latent_planner.datasets.d4rl_utils import qlearning_dataset_with_timeouts

    config = DefaultConfig()
    config.env_name = "antmaze-large-play-v2"
    config.dataset = None

    seq_len = config.subsampled_seq_len * config.step
    data_config = DatasetConfig(
        save_dir=(config.save_dir, 'data_config.pkl'),
        env_name=config.env_name,
        dataset=config.dataset,
        termination_penalty=-100,
        seq_len=seq_len,
        step=config.step,
        discount=config.discount,
        disable_goal=config.disable_goal,
        normalize_sa=config.normalize_state,
        normalize_reward=config.normalize_reward,
        max_path_length=1001,
        min_path_length=3,
        device=config.device
    )

    import matplotlib.pyplot as plt

    dataset = SeqDataset(data_config)

    plt.plot(dataset.rewards[0, :, 0]);plt.show()
    plt.plot(dataset.values[0]);plt.show()
    plt.plot(dataset.rewards[471, :, 0]);plt.show()
    plt.plot(dataset.values[471]);plt.show()
    plt.plot(dataset.rewards[624, :, 0]);plt.show()
    plt.plot(dataset.values[624]);plt.show()

    print(dataset[0])
    print(dataset[999])
    print(dataset[1000])
    print(dataset[1001])