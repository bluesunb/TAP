import os, time
import numpy as np
import gym
import random
import h5py
from tqdm import tqdm

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
    ## d4rl prints out a variety of warnings
    # import d4rl

# def construct_dataloader(dataset, **kwargs):
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
#     return dataloader

def minrl_dataset(dataset):
    #dataset = dict(observations=dataset[0]['vector'], actions=dataset[1]['vector'])
    obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []


    trajectory_names = dataset.get_trajectory_names()
    random.shuffle(trajectory_names)

    for trajectory_name in trajectory_names:
        data_gen = dataset.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for obs, action, reward, new_obs, done in data_gen:
            obs = obs['pov'].flatten()
            action = action['vector']

            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done)
            realdone_.append(done)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'rewards': np.array(reward_)[:, None],
        'terminals': np.array(done_)[:, None],
        'terminals_float': np.array(realdone_)[:, None],
    }


def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=True, disable_goal=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate([dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            dataset["observations"] = np.concatenate([dataset["observations"], np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)],
                                                     axis=1)

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = (dataset['infos/goal'][i] != dataset['infos/goal'][i+1]).any()
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'terminals_float': np.array(realdone_)[:,None],
    }


def get_trajectories(env_name: str,
                     dataset: str = None,
                     terminate_on_end: bool = True,
                     disable_goal: bool = False):
    d4rl_dataset_dir = os.path.expanduser('~/.d4rl/datasets')
    if dataset:
        dataset_dir = os.path.join(d4rl_dataset_dir, dataset + '.hdf5')
        if os.path.exists(dataset_dir):
            print(f'[ datasets/sequence ] Loading {dataset_dir}')
            dataset = get_dataset(dataset_dir)
    else:
        with suppress_output():
            import d4rl
        env: d4rl.offline_env.OfflineEnv = load_environment(env_name) if isinstance(env_name, str) else env_name
        dataset = env.get_dataset()

    length = dataset['observations'].shape[0]
    if "infos/goal" in dataset:
        goal = np.zeros(length, 2, dtype=np.float32) if disable_goal else dataset['infos/goal']
        dataset["observations"] = np.concatenate([dataset["observations"], goal], axis=1)
        env_reset = np.any(dataset['infos/goal'][:-1] != dataset['infos/goal'][1:], axis=1)
    else:
        env_reset = dataset['timeouts'][:-1]

    observations = dataset['observations'][:-1]
    next_observations = dataset['observations'][1:]
    actions = dataset['actions'][:-1]
    rewards = dataset['rewards'][:-1]
    dones = dataset['terminals'][:-1]
    truncated = env_reset
    # terminals = np.logical_or(dataset['terminals'][:-1], env_reset)
    # timeouts = dataset['terminals'][:-1]
    # timeouts = dataset['timeouts'][:-1]

    if not terminate_on_end:
        observations = observations[~env_reset]
        next_observations = next_observations[~env_reset]
        actions = actions[~env_reset]
        rewards = rewards[~env_reset]
        dones = dones[~env_reset]
        truncated = truncated[~env_reset]

    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'rewards': rewards[:, None],
        'terminals': dones[:, None],
        'truncated': truncated[:, None],
    }


class MineRLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return obs['pov'].flatten()

class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return {'vector': action}

    def state_vector(self):
        return np.zeros([1])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps

    env.name = name
    return env


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


# def get_dataset(h5path):
#     data_dict = {}
#     with h5py.File(h5path, 'r') as dataset_file:
#         for k in tqdm(get_keys(dataset_file), desc="load datafile"):
#             try:  # first try loading as an array
#                 data_dict[k] = dataset_file[k][:]
#             except ValueError as e:  # try loading as a scalar
#                 data_dict[k] = dataset_file[k][()]
#
#     # Run a few quick sanity checks
#     for key in ['observations', 'actions', 'rewards', 'terminals']:
#         assert key in data_dict, 'Dataset is missing key %s' % key
#     N_samples = data_dict['observations'].shape[0]
#     if data_dict['rewards'].shape == (N_samples, 1):
#         data_dict['rewards'] = data_dict['rewards'][:, 0]
#     assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
#         str(data_dict['rewards'].shape))
#     if data_dict['terminals'].shape == (N_samples, 1):
#         data_dict['terminals'] = data_dict['terminals'][:, 0]
#     assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
#         str(data_dict['rewards'].shape))
#     return data_dict

def make_env(env_name: str):
    import d4rl
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                disable_goal=False,
                obs_dtype=np.float32,):

    if dataset is None:
        # dataset = d4rl.qlearning_dataset(env)
        # dataset = qlearning_dataset_with_timeouts(env)
        dataset = get_trajectories(env_name, dataset=dataset, terminate_on_end=True, disable_goal=disable_goal)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    dataset['terminals'][-1] = 1
    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset['terminals'])[0]
        last_idx = np.nonzero(dataset['terminals'])[0]
        penult_idx = last_idx - 1
        new_dataset = dict()
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    if 'antmaze' in env_name:
        dataset['rewards'] = dataset['rewards'] - 1
    #     # antmaze: terminals are incorrect for GCRL
    #     dones_float = np.zeros_like(dataset['terminals']).squeeze()
    #     dataset['terminals'][:] = 0.
    #
    #     norms = np.linalg.norm(dataset['observations'][1:] - dataset['next_observations'][:-1], axis=-1)
    #     norms = (norms > 1e-6).astype(np.float32)
    #     dones_float[:-1] = norms
    #     dones_float[-1] = 1
    #     dones_float = dones_float.reshape(*dataset['terminals'].shape)
    # else:
    #     dones_float = dataset['terminals'].copy()
    dones_float = dataset['terminals'].copy()

    observations = dataset['observations'].astype(obs_dtype)
    next_observations = dataset['next_observations'].astype(obs_dtype)

    return {'observations': observations,
            'actions': dataset['actions'].astype(np.float32),
            'rewards': dataset['rewards'].astype(np.float32),
            'masks': 1.0 - dones_float.astype(np.float32),
            'terminals': dones_float.astype(bool),
            'truncated': dataset['truncated'].astype(bool),
            'next_observations': next_observations.astype(np.float32)}


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
