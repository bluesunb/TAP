import gym
from collections import defaultdict
from latent_planner.config import DatasetConfig
from latent_planner.datasets.seq import SeqDataset
from typing import List, Dict, Sequence


def visualize(env, episode: Dict[str, Sequence]):

    env.reset()
    context = defaultdict(list)

    frames = [env.render(mode='rgb_array')]
    env.viewer.cam.lookat[0] = 18
    env.viewer.cam.lookat[1] = 12
    env.viewer.cam.distance = 50
    env.viewer.cam.elevation = -90

    for t in range(env.max_episode_steps):
        action = episode['actions'][t]
        obs, reward, done, info = env.step(action)
        frames.append(env.render(mode='rgb_array'))

        context['observations'].append(obs)
        context['actions'].append(action)
        context['rewards'].append(reward)
        context['dones'].append(done)
        context['info/goal'].append(info['goal'])