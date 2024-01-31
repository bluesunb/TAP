import numpy as np
import torch as th
from collections import defaultdict
from latent_planner.models.autoencoders import VQContinuousVAE, TransformerPrior

from typing import List, Tuple, Dict, Any, Callable, Optional


def denormalize_rv(reward: th.Tensor,
                   value: th.Tensor,
                   denormalize_rew: Callable[[th.Tensor], th.Tensor],
                   denormalize_val: Callable[[th.Tensor], th.Tensor],
                   shape: Tuple[int, int]):

    if denormalize_rew is not None:
        reward = denormalize_rew(reward).view(*shape)
    if denormalize_val is not None:
        value = denormalize_val(value).view(*shape)

    return reward, value


@th.no_grad()
def model_rollout_continuous(model: VQContinuousVAE,
                             x: th.Tensor,
                             latent: th.Tensor,
                             denormalize_rew: Callable[[th.Tensor], th.Tensor],
                             denormalize_val: Callable[[th.Tensor], th.Tensor],
                             discount: float = 0.99,
                             prob_penalty_weight: float = 1e-4):
    """
    Predicts the trajectory using latent conditioned on last transition in x, and computes the resultant objective.

    Returns:
        - pred: (bs * seq_len, transition_dim): predicted transitions
        - objective: (bs,): objective for each trajectory
    """

    bs, seq_len, _ = x.shape
    pred = model.decode(latent, condition=x[:, -1, :model.config.observation_dim])  # (bs, seq_len, trans_dim)
    pred = pred.view(-1, model.config.transition_dim)        # (bs * seq_len, trans_dim)

    r = pred[:, 2]
    v = pred[:, 3]
    r, v = denormalize_rv(r, v, denormalize_rew, denormalize_val, (bs, seq_len))

    terminal = pred[:, -1].view(bs, seq_len)
    disc_reward = th.cumprod(th.ones_like(r) * discount * (1 - terminal), dim=-1)
    values = th.sum(r[:, :-1] * disc_reward[:, :-1], dim=-1) + v[:, -1] * disc_reward[:, -1]
    prob_penalty = prob_penalty_weight * th.mean(th.square(latent), dim=-1)
    objective = values - prob_penalty
    return pred.cpu().numpy(), objective.cpu().numpy()


@th.no_grad()
def sample_uniform(model: VQContinuousVAE,
                   x: th.Tensor,
                   denormalize_rew: Callable[[th.Tensor], th.Tensor],
                   denormalize_val: Callable[[th.Tensor], th.Tensor],
                   discount: float = 0.99,
                   steps: int = 100,
                   nb_samples: int = 4095):

    indices = th.randint(0, model.config.n_tokens - 1, size=(nb_samples, steps // model.config.latent_step))
    indices = indices.to(x.device)
    bs, seq_len = indices.shape

    pred = model.decode_from_indices(indices, x[:, 0, :model.config.observation_dim])
    r = pred[..., 2]
    v = pred[..., 3]
    r, v = denormalize_rv(r, v, denormalize_rew, denormalize_val, (bs, seq_len))

    disc_reward = th.cumprod(th.ones_like(r) * discount, dim=-1)
    values = th.sum(r[:, :-1] * disc_reward[:, :-1], dim=-1) + v[:, -1] * disc_reward[:, -1]
    optimal = pred[th.argmax(values), ...]
    return optimal.cpu().numpy()


@th.no_grad()
def sample_with_prior(model: VQContinuousVAE,
                      prior: TransformerPrior,
                      x: th.Tensor,
                      denormalize_rew: Callable[[th.Tensor], th.Tensor],
                      denormalize_val: Callable[[th.Tensor], th.Tensor],
                      discount: float = 0.99,
                      steps: int = 100,
                      nb_samples: int = 4096,
                      n_iter: int = 8,
                      likelihood_weight: float = 0.05,
                      prob_threshold: float = 0.05,
                      uniform: bool = False,
                      return_info: bool = False):

    obs = x[:, 0, :model.config.observation_dim]
    optimal_trajs = []
    optimal_values = []
    info = defaultdict(list)

    for i in range(n_iter):
        indices = None      # (bs, seq_len // latent_step)
        log_joint_probs = th.zeros(1).to(x)

        n_steps = steps // model.config.latent_step
        for step in range(n_steps):
            logits, loss = prior(indices, condition=obs)
            probs = th.softmax(logits[:, -1, :], dim=-1)    # (bs, n_tokens) : probability of new predicted token
            if uniform:
                probs = (probs > 0) / (probs > 0).sum(dim=-1)[:, None]

            samples = th.multinomial(probs,
                                     num_samples=1 if step != 0 else nb_samples // n_iter,
                                     replacement=True)    # (bs, n_samples)
            # samples_prob = th.stack([th.index_select(log_p, 0, k) for log_p, k in zip(probs.log(), samples)])   # (bs, n_samples)
            samples_log_prob = th.gather(probs.log(), dim=-1, index=samples)   # (bs, n_samples)
            log_joint_probs += samples_log_prob.reshape(-1)   # (bs * n_samples, )

            if indices is not None:
                indices = th.cat([indices, samples.view(-1, 1)], dim=1)
            else:
                indices = samples.view(-1, step + 1)

        pred = model.decode_from_indices(indices, obs)  # (bs, seq_len, trans_dim)
        r = pred[..., 2]
        v = pred[..., 3]
        r, v = denormalize_rv(r, v, denormalize_rew, denormalize_val, (indices.shape[0], -1))

        disc_reward = th.cumprod(th.ones_like(r) * discount, dim=-1)
        values = th.sum(r[:, :-1] * disc_reward[:, :-1], dim=-1) + v[:, -1] * disc_reward[:, -1]
        likelihood_bonus = likelihood_weight * th.clamp(log_joint_probs, -1e5, np.log(prob_threshold) * n_steps)    # < 0
        objective = values + likelihood_bonus

        if return_info:
            info['log_probs'].append(log_joint_probs.cpu().numpy())
            info['returns'].append(values.cpu().numpy())
            info['predictions'].append(pred.cpu().numpy())
            info['objectives'].append(objective.cpu().numpy())
            info['latent_codes'].append(indices.cpu().numpy())

        max_idx = th.argmax(objective)
        optimal_values.append(values[max_idx].item())
        optimal_trajs.append(pred[max_idx, ...])

    if return_info:
        for k, v in info.items():
            info[k] = np.concatenate(v, axis=0)

    max_idx = np.array(optimal_values).argmax()
    optimal_traj = optimal_trajs[max_idx]
    print(f"[sample/prior] predicted max value: {optimal_values[max_idx]:.5f}")

    if return_info:
        return optimal_traj.cpu().numpy(), info
    return optimal_traj.cpu().numpy()


@th.no_grad()
def sample_with_prior_tree(model: VQContinuousVAE,
                           prior: TransformerPrior,
                           x: th.Tensor,
                           denormalize_rew: Callable[[th.Tensor], th.Tensor],
                           denormalize_val: Callable[[th.Tensor], th.Tensor],
                           discount: float = 0.99,
                           steps: int = 100,
                           samples_per_latent: int = 16,
                           likelihood_weight: float = 0.05):

    indices = None
    obs = x[:, 0, :model.config.observation_dim]
    joint_probs = th.ones(1).to(x)

    n_steps = steps // model.config.latent_step
    for step in range(n_steps):
        logits, loss = prior(indices, condition=obs)
        probs = th.softmax(logits[:, -1, :], dim=-1)    # (bs, n_tokens) : probability of new predicted token
        samples = th.multinomial(probs, num_samples=samples_per_latent, replacement=True)   # (bs, n_samples)
        samples_prob = th.gather(probs, dim=-1, index=samples)
        joint_probs = th.repeat_interleave(joint_probs, samples_per_latent, dim=0) * samples_prob.view(-1)

        if indices is not None:
            indices = th.cat([th.repeat_interleave(indices, samples_per_latent, dim=0), samples.view(-1, 1)],
                             dim=-1)
        else:
            indices = samples.view(-1, step + 1)

    pred = model.decode_from_indices(indices, condition=obs)
    r = pred[..., 2]
    v = pred[..., 3]
    r, v = denormalize_rv(r, v, denormalize_rew, denormalize_val, (indices.shape[0], -1))

    disc_reward = th.cumprod(th.ones_like(r) * discount, dim=-1)
    values = th.sum(r[:, :-1] * disc_reward[:, :-1], dim=-1) + v[:, -1] * disc_reward[:, -1]
    likelihood_bonus = likelihood_weight * th.clamp(joint_probs.log(), -1e5, None)
    objective = values + likelihood_bonus

    max_idx = th.argmax(objective)
    optimal_traj = pred[max_idx, ...]
    print(f"[sample/prior] predicted max value: {values[max_idx]:.5f} | likelihood: {joint_probs[max_idx]:4f} |"
          f"ood-obj: {likelihood_bonus[max_idx]:.5f}")

    return optimal_traj.cpu().numpy()


@th.no_grad()
def beam_with_prior(model: VQContinuousVAE,
                    prior: TransformerPrior,
                    x: th.Tensor,
                    denormalize_rew: Callable[[th.Tensor], th.Tensor],
                    denormalize_val: Callable[[th.Tensor], th.Tensor],
                    discount: float = 0.99,
                    steps: int = 100,
                    beam_width: int = 10,
                    n_expand: int = 10,
                    likelihood_weight: float = 0.05,
                    prob_threshold: float = 0.05,
                    cum_method: str = 'product',
                    return_info: bool = False):

    assert cum_method in ('product', 'min', 'expect'), f"cum_method must be one of 'product', 'min', 'expect'"

    indices = None      # (nb_samples, ) -> (beam_width, step)
    obs = x[:, 0, :model.config.observation_dim]
    joint_log_probs = th.zeros(1).to(x)     # (nb_samples, ) -> (beam_width, )
    info = {}

    n_steps = steps // model.config.latent_step
    for step in range(n_steps):
        logits, loss = prior(indices, condition=obs)    # (1, step + 1, n_tokens)
        probs = th.softmax(logits[:, -1, :], dim=-1)    # (1, n_tokens) : probability of new predicted token
        nb_samples = (beam_width * n_expand) if step == 0 else n_expand
        samples = th.multinomial(probs, num_samples=nb_samples, replacement=True)   # (1, n_samples)
        samples_log_prob = th.gather(probs.log(), dim=-1, index=samples)    # (1, n_samples)

        if cum_method in ('product', 'expect'):
            joint_log_probs = th.repeat_interleave(joint_log_probs, nb_samples, dim=0) + samples_log_prob.view(-1)
        elif cum_method == 'min':
            joint_log_probs = th.minimum(th.repeat_interleave(joint_log_probs, nb_samples, dim=0),
                                         samples_log_prob.view(-1))

        if indices is not None:
            indices = th.cat([th.repeat_interleave(indices, nb_samples, dim=0), samples.view(-1, 1)],
                             dim=1)
        else:
            indices = samples.view(-1, step + 1)    # (nb_samples, step + 1)

        pred = model.decode_from_indices(indices, condition=obs)    # (nb_samples, (step + 1) * latent_step, trans_dim)
        r = pred[..., 2]
        v = pred[..., 3]
        r, v = denormalize_rv(r, v, denormalize_rew, denormalize_val, (indices.shape[0], -1))   # (nb_samples, (step + 1) * latent_step)

        disc_reward = th.cumprod(th.ones_like(r) * discount, dim=-1)   # (nb_samples, (step + 1) * latent_step)
        values = th.sum(r[:, :-1] * disc_reward[:, :-1], dim=-1) + v[:, -1] * disc_reward[:, -1]  # (nb_samples, )

        nb_top = beam_width if step < n_steps - 1 else 1

        if cum_method == 'product':
            likelihood_bonus = th.clamp(joint_log_probs, -1e5, np.log(prob_threshold) * n_steps) * likelihood_weight
        else:   # cum_method == 'min'
            likelihood_bonus = th.clamp(joint_log_probs, 0, np.log(prob_threshold)) * likelihood_weight

        objective = values + likelihood_bonus
        if cum_method == 'expect':
            sample_objective = values * th.exp(joint_log_probs)
        else:
            sample_objective = objective

        topk_values, topk_idx = th.topk(sample_objective, k=nb_top, dim=-1)

        if return_info:
            info[(step + 1) * model.config.latent_step] = {
                'predictions': pred.cpu().numpy(),
                'returns': values.cpu().numpy(),
                'latent_codes': indices.cpu().numpy(),
                'log_probs': joint_log_probs.cpu().numpy(),
                'objectives': objective.cpu().numpy(),
                'top_k_idx': topk_idx.cpu().numpy(),
            }

        indices = indices[topk_idx]
        joint_log_probs = joint_log_probs[topk_idx]

    optimal_traj = pred[topk_idx[0], ...]
    print(f"[beam/prior] predicted max value: {topk_values[0]:.5f} | likelihood: {joint_log_probs[0]:4f} |"
          f"ood-obj: {likelihood_bonus[topk_idx[0]]:.5f}")

    if return_info:
        return optimal_traj.cpu().numpy(), info
    return optimal_traj.cpu().numpy()
