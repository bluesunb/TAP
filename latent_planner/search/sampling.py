import numpy as np
import torch as th
import torch.nn as nn

import latent_planner.search.search_utils as utils


def forward(model: nn.Module,
            x: th.Tensor,
            returnx: bool = False,
            max_block: int = np.inf,
            allow_crop: bool = True,
            crop_increment: int = None,
            **kwargs):

    model.eval()
    block_size = min(model.config.block_size, max_block)

    if x.shape[1] > block_size:
        assert allow_crop, f"Input sequence length {x.shape[1]} is larger than the maximum block size {block_size}."
        n_crop = utils.round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    if returnx:
        logits, _, output = model(x, returnx=returnx, **kwargs)
        return logits, output

    else:
        logits, _ = model(x, returnx=returnx, **kwargs)
        return logits


def forward_continuous(model: nn.Module,
                       x: th.Tensor,
                       returnx: bool = False,
                       max_block: int = np.inf,
                       allow_crop: bool = True,
                       crop_increment: int = None,
                       **kwargs):

    model.eval()
    block_size = min(model.config.block_size, max_block)

    if x.shape[1] > block_size:
        assert allow_crop, f"Input sequence length {x.shape[1]} is larger than the maximum block size {block_size}."
        n_crop = utils.round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    if returnx:
        logits, _, output = model(x, **kwargs, returnx=returnx)
        return logits, output


    else:
        state_pred, action_pred, reward_pred, value_pred, action_prob, _ = model(x, **kwargs, returnx=returnx)

        action_dist = th.distributions.categorical.Categorical(action_prob)
        action_idx = action_dist.sample()
        action_idx = action_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, action_pred.shape[-1])
        action_sample = th.gather(action_pred, -2, action_idx).squeeze(-2)

        logits = th.cat([state_pred, action_sample, reward_pred, value_pred], dim=-1)
        return logits, action_prob, action_pred


def get_log_p(model: nn.Module,
              x: th.Tensor,
              temperature: float=1.0,
              topk: int=0,
              threshold: float=0,
              **kwargs):
    logits = forward(model, x, **kwargs)
    logits = logits[:, -1] / temperature

    if threshold > 0:
        logits = utils.filter_cdf(logits, threshold)
    if topk > 0:
        logits = utils.top_k_logits(logits, topk)

    return logits.log_softmax(dim=-1)


def sample(model: nn.Module,
           x: th.Tensor,
           returnx: bool=False,
           temperature: float=1.0,
           topk: int=0,
           threshold: float=0,
           **kwargs):
    if returnx:
        logits, output = forward(model, x, returnx=True, **kwargs)
    else:
        logits = forward(model, x, returnx=False, **kwargs)

    logits = logits[:, -1] / temperature
    raw_probs = logits.softmax(dim=-1)

    if threshold > 0:
        logits = utils.filter_cdf(logits, threshold)
    if topk > 0:
        logits = utils.top_k_logits(logits, topk)

    probs = logits.softmax(dim=-1)
    indices = th.multinomial(probs, num_samples=1)

    if returnx:
        return indices, raw_probs, output
    else:
        return indices, raw_probs


def sample_continuous(model: nn.Module,
                      x: th.Tensor,
                      idx_start: int,
                      idx_end: int,
                      returnx: bool=False,
                      **kwargs):
    logits, action_prob, action_pred = forward_continuous(model, x, returnx=True, **kwargs)
    logits = logits[:, -1, idx_start:idx_end]
    action_prob = action_prob[:, -1]
    action_pred = action_pred[:, -1]
    return logits, action_prob, action_pred


def sample_n(model: nn.Module,
             x: th.Tensor,
             N: int,
             returnx: bool=False,
             **kwargs):

    batch_size = len(x)
    probs = th.zeros(batch_size, N, model.config.vocab_size + 1, device=x.device)

    for n in range(N):
        if returnx:
            indices, p, output = sample(model, x, returnx, **kwargs)
        else:
            indices, p = sample(model, x, **kwargs)

        x = th.cat((x, indices), dim=1)
        probs[:, n] = p

    if returnx:
        return x, probs, output
    else:
        return x, probs


@th.no_grad()
def sample_n_continuous(model, x, N, start_idx, returnx=False, **sample_kwargs):
    batch_size, t, _ = x.size()

    if start_idx == 0:
        input_x = x[:, :-1]
    else:
        input_x = x

    if returnx:
        logits, output = sample_continuous(model, input_x, start_idx, start_idx + N, returnx, **sample_kwargs)
    else:
        logits, action_prob, action_pred = sample_continuous(model, input_x, start_idx, start_idx + N, returnx,
                                                             **sample_kwargs)
    x[:, -1, start_idx:start_idx + N] = logits

    if returnx:
        return x, output
    else:
        return x, action_prob, action_pred