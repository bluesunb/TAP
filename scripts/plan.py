import collections
import json
import os, pickle
import numpy as np
import torch as th

from latent_planner.config import (DefaultConfig, PlannerConfig,
                                   DatasetConfig, TransformerConfig, TrainerConfig, load_config, get_recent)
from latent_planner.datasets.seq import SeqDataset
from latent_planner.datasets.dataset_utils import normalize, denormalize, denormalize_joined
from latent_planner.datasets.preprocessing import get_preprocess_fn
from latent_planner.models.autoencoders import VQContinuousVAE, TransformerPrior, VQContinuousVAEEncWrap
from latent_planner.search import search_utils as utils
from latent_planner.search import optimizer as plan_opt
from latent_planner.train_utils.timer import Timer


def load_environment(name):
    import gym, d4rl
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def prepare(env_name, dataset_name=None, test_planner='sample_prior', task_type='locomotion'):
    config = DefaultConfig()
    config.env_name = env_name
    config.dataset = dataset_name
    config.task_type = task_type

    save_dir = get_recent(config.save_dir)
    planner_config = PlannerConfig(save_dir=(save_dir, 'planner_config.pkl'),
                                   test_planner=test_planner,
                                   task_type=config.task_type)

    dataset_config = load_config(os.path.join(save_dir, 'data_config.pkl'), DatasetConfig)
    model_config = load_config(os.path.join(save_dir, 'model_config.pkl'), TransformerConfig)
    prior_config = load_config(os.path.join(save_dir, 'prior_model_config.pkl'), TransformerConfig)

    # ============= Dataset =============
    dataset = SeqDataset(dataset_config)

    # ============= Model =============
    # model_name = max(list(filter(lambda x: 'state' in x, os.listdir(config.save_dir))),
    #                  key=lambda x: int(x.split('_')[1].split('.')[0]))
    model_name = 'state_12.pt'
    model = VQContinuousVAE(model_config)
    state_dict = th.load(os.path.join(save_dir, model_name))
    state_dict = collections.OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    # model = model.load_state_dict(th.load(os.path.join(config.save_dir, model_name)))
    model.load_state_dict(state_dict)
    print(f'Loaded model from {os.path.join(save_dir, model_name)}')
    model.to(config.device)
    if config.normalize_state:
        padding_vector = dataset.normalize_joined_single(np.zeros(model_config.transition_dim - 1))
        model.padding_vector = th.from_numpy(padding_vector).to(model.padding_vector)

    # prior_name = max(list(filter(lambda x: 'prior' in x, os.listdir(config.save_dir))),
    #                  key=lambda x: int(x.split('_')[1].split('.')[0]))
    prior_name = 'prior_51.pt'
    prior = TransformerPrior(prior_config)
    state_dict = th.load(os.path.join(save_dir, prior_name))
    state_dict = collections.OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    prior.load_state_dict(state_dict)
    # prior.load_state_dict(th.load(os.path.join(config.save_dir, prior_name)))
    print(f'Loaded prior from {os.path.join(save_dir, prior_name)}')
    prior.to(config.device)

    configs = {'all': config,
               'planner': planner_config,
               'dataset': dataset_config,
               'model': model_config,
               'prior': prior_config,
               'save_dir': save_dir}

    preprocess_fn = get_preprocess_fn(env_name)

    return dataset, model.eval(), prior.eval(), preprocess_fn, configs


def main(env_name, dataset_name=None, test_planner='sample_prior', task_type='locomotion'):
    dataset, model, prior, preprocess_fn, configs = prepare(env_name, dataset_name, test_planner, task_type)
    config: DefaultConfig = configs['all']
    planner_config: PlannerConfig = configs['planner']
    dataset_config: DatasetConfig = configs['dataset']
    model_config: TransformerConfig = configs['model']
    prior_config: TransformerConfig = configs['prior']

    reward_dim = value_dim = 1
    transition_dim = model_config.transition_dim
    timer = Timer()

    env = load_environment(env_name)
    obs = env.reset()
    total_reward = 0
    discount_reward = 0

    rollout = []
    if "antmaze" in env_name:
        obs = np.concatenate([obs, np.zeros(2) if dataset_config.disable_goal else env.target_goal])
        rollout.append(
            np.concatenate([env.state_vector().copy(), np.zeros(2) if dataset_config.disable_goal else env.target_goal])
        )
    else:
        rollout.append(np.concatenate([env.state_vector().copy()]))

    context = []
    losses = []
    frames = [env.render(mode='rgb_array')]
    env.viewer.cam.lookat[0] = 18
    env.viewer.cam.lookat[1] = 12
    env.viewer.cam.distance = 50
    env.viewer.cam.elevation = -90

    model.eval()
    for t in range(env.max_episode_steps):
        obs = preprocess_fn(obs)
        state = env.state_vector()

        if dataset_config.normalize_sa:
            # obs = normalize(obs, dataset.obs_mean, dataset.obs_std)
            obs = (obs - dataset.obs_mean.squeeze()) / dataset.obs_std.squeeze()

        if "antmaze" in env.name:
            state = np.concatenate([state, np.zeros(2) if dataset_config.disable_goal else env.target_goal])

        if t % planner_config.plan_freq == 0:
            prefix = utils.make_prefix(obs, transition_dim, device=config.device)
            prefix = prefix[-1, -1, None, None]

            if planner_config.test_planner == "beam_with_prior":
                prior.eval()
                sequence = plan_opt.beam_with_prior(
                    model, prior, prefix,
                    denormalize_rew=lambda x: denormalize(x, dataset.rew_mean, dataset.rew_std),
                    denormalize_val=lambda x: denormalize(x, dataset.value_mean, dataset.value_std),
                    steps=int(planner_config.horizon),
                    beam_width=planner_config.beam_width,
                    n_expand=planner_config.n_expend,
                    likelihood_weight=planner_config.prob_weight,
                    prob_threshold=float(planner_config.prob_threshold),
                    discount=dataset_config.discount,
                )

            elif planner_config.test_planner == "sample_prior":
                prior.eval()
                sequence = plan_opt.sample_with_prior(
                    model, prior, prefix,
                    denormalize_rew=lambda x: denormalize(x, dataset.rew_mean, dataset.rew_std),
                    denormalize_val=lambda x: denormalize(x, dataset.value_mean, dataset.value_std),
                    steps=int(planner_config.horizon),
                    nb_samples=planner_config.nb_samples,
                    n_iter=planner_config.rounds,
                    prob_threshold=float(planner_config.prob_threshold),
                    likelihood_weight=planner_config.prob_weight,
                    uniform=planner_config.uniform,
                    discount=dataset_config.discount,
                )
            elif planner_config.test_planner == "sample_prior_tree":
                prior.eval()
                sequence = plan_opt.sample_with_prior_tree(
                    model, prior, prefix,
                    denormalize_rew=lambda x: denormalize(x, dataset.rew_mean, dataset.rew_std),
                    denormalize_val=lambda x: denormalize(x, dataset.value_mean, dataset.value_std),
                    steps=int(planner_config.horizon),
                    nb_samples=planner_config.nb_samples,
                    discount=dataset_config.discount
                )
            else:
                raise NotImplementedError('Planner not implemented')
        else:
            sequence = sequence[:, 1:]

        if t == 0:
            first_value = float(denormalize(sequence[0, -2], dataset.value_mean, dataset.value_std))
            first_search_value = float(denormalize(sequence[-1, -2], dataset.value_mean, dataset.value_std))

        print(f"Step {t}: {sequence.shape[0]} plans")
        print(denormalize(sequence[0, -2], dataset.value_mean, dataset.value_std))

        sequence_recon = sequence
        action = utils.extract_actions(sequence_recon, model_config.observation_dim, model_config.action_dim, t=0)
        if dataset_config.normalize_sa:
            action = denormalize(action, dataset.act_mean, dataset.act_std)
            sequence_recon = denormalize_joined(sequence_recon, dataset.mean, dataset.std)

        next_obs, reward, term, _ = env.step(action)

        if "antmaze" in env.name:
            next_obs = np.concatenate([next_obs, np.zeros(2) if dataset_config.disable_goal else env.target_goal])

        total_reward += reward
        discount_reward += reward * dataset_config.discount ** t
        score = env.get_normalized_score(total_reward)

        rollout.append(state.copy())
        context = utils.update_context(obs, action, reward, config.device)

        print(
            f'[ plan ] t: {t} / {env.max_episode_steps} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
            f'time: {timer():.4f} | {config.dataset} | {config.exp_name} | {config.suffix}\n'
        )

        frames.append(env.render(mode='rgb_array'))
        if term:
            break

        obs = next_obs

    import imageio
    imageio.mimsave(os.path.join(configs['save_dir'], f'{env_name}.mp4'), frames, fps=30)

    print(f"score: {score:.4f} | total_reward: {total_reward:.4f} | discount_reward: {discount_reward:.4f}")
    print(f"first_value: {first_value:.4f} | first_search_value: {first_search_value:.4f}")


if __name__ == '__main__':
    env_name = "antmaze-large-diverse-v2"
    dataset_name = None
    task_type = "locomotion"
    test_planner = "beam_with_prior"
    main(env_name, dataset_name, test_planner, task_type)