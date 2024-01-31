import numpy as np
import torch as th

from latent_planner.search import search_utils as utils
from latent_planner.search.sampling import sample_n, get_log_p

REWARD_DIM = VALUE_DIM = 1


@th.no_grad()
def beam_plan(
        model, value_fn, x,
        n_steps, beam_width, n_expand,
        observation_dim, action_dim,
        discount=0.99, max_context_transitions=None,
        k_obs=None, k_act=None, k_rew=1,
        cdf_obs=None, cdf_act=None, cdf_rew=None,
        verbose=True, previous_actions=None,
        return_all=False,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''
    inp = x.clone()

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    # pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    # repeat input for search
    x = x.repeat(beam_width, 1)

    # construct reward and discount tensors for estimating values
    rewards = th.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** th.arange(n_steps + 1, device=x.device)

    # logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    for t in range(n_steps):
        # repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        # sample actions
        x, p = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        # prob = th.gather(p, dim=-1, index=x[:, -6:].unsqueeze(-1))
        # print(prob.squeeze())
        # exit()

        # sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        # optionally, use a percentile or mean of the reward and
        # value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)

        # update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t + 1] = V_t

        # estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        # get `beam_width` best actions
        values, inds = th.topk(values, beam_width)

        # index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        # sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        # logging
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            'discount': discount
        })

    progress.stamp()

    # [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)

    # crop out context transitions
    # [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    # return best sequence
    argmax = values.argmax()
    best_sequence = x[argmax]

    if return_all:
        return best_sequence, x, values
    else:
        return best_sequence


@th.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = th.arange(len(x), dtype=th.long, device=x.device)
    cumulative_logp = th.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):

        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_log_p(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_log_p(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = utils.sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = th.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()