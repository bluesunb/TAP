import os
import pickle
import gym, d4rl
import imageio

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from latent_planner.models.autoencoders import VQContinuousVAE, TransformerPrior
from latent_planner.config import (DefaultConfig, DatasetConfig, TransformerConfig, TrainerConfig, load_config)


def prepare(env_name, exp_name):
    # env_name = "maze2d-medium-v1"
    # exp_name = 'vqvae'

    config = DefaultConfig()
    config.env_name = env_name
    config.exp_name = exp_name

    save_dir = config.save_dir
    dataset_config = pickle.load(open(os.path.join(save_dir, 'data_config.pkl'), 'rb'))
    model_config = pickle.load(open(os.path.join(save_dir, 'model_config.pkl'), 'rb'))
    prior_config = pickle.load(open(os.path.join(save_dir, 'prior_model_config.pkl'), 'rb'))
    trainer_config = pickle.load(open(os.path.join(save_dir, 'trainer_config.pkl'), 'rb'))

    model = VQContinuousVAE(model_config)
    model.load_state_dict(th.load(os.path.join(save_dir, 'model.pt'), map_location='cpu'))
    model.eval()

    prior = TransformerPrior(prior_config)
    prior.load_state_dict(th.load(os.path.join(save_dir, 'prior.pt'), map_location='cpu'))
    prior.eval()

    configs = {'all': config,
               'dataset_config': dataset_config,
               'model_config': model_config,
               'prior_config': prior_config,
               'trainer_config': trainer_config}

    return model, prior, configs


def evaluate(model, prior,
             configs: dict,
             data_config: DatasetConfig,
             model_config: TransformerConfig,
             save_name: str = 'eval'):

    model.eval()
    prior.eval()

    config: DefaultConfig = configs['all']
    dataset_config: DatasetConfig = configs['dataset_config']
    model_config: TransformerConfig = configs['model_config']
    prior_config: TransformerConfig = configs['prior_config']
    trainer_config: TrainerConfig = configs['trainer_config']

    save_dir = os.path.join(config.save_dir, save_name)

    env = gym.make(config.env_name)
    env.seed(0)

    state = env.reset()
    done = False
    frames = [env.render(mode='rgb_array')]

    while not done:
        action = model