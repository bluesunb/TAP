import time
import math
import re
import numpy as np
import torch as th

DTYPE = th.float
DEVICE = 'cuda:0'
VALUE_PLACEHOLDER = 1e6


def top_k_logits(logits: th.Tensor, k: int):
    vals, ids = th.topk(logits, k)
    out = logits.clone()
    out[out < vals[..., -1:]] = -float('Inf')  # mask out all logits that are not in the top k
    return out


# def filter_cdf_prob(probs: th.Tensor, threshold: float):
#     """
#     Filter out all probabilities that are below the threshold.
#     """
#     batch_ids = th.arange(probs.shape[0], device=probs.device)
#     bins_ids = th.arange(probs.shape[1], device=probs.device)
#
#     probs_sorted, idx = th.sort(probs, dim=-1, descending=False)
#     cum_probs = th.cumsum(probs_sorted, dim=-1)
#
#     mask = cum_probs < threshold
#     masked_ids = th.argmax(mask * bins_ids, dim=-1)
#     probs_threshold = probs_sorted[batch_ids, masked_ids]
#
#     out = probs.clone()
#     probs_mask = probs <= probs_threshold.unsqueeze(-1)
#     out[probs_mask] = 1e-8
#     return out


def filter_prob(probs: th.Tensor, threshold: float):
    probs_sorted, idx = th.sort(probs, dim=-1, descending=False)
    cum_probs = th.cumsum(probs_sorted, dim=-1)

    masked_ids = th.searchsorted(cum_probs, th.full((cum_probs.shape[0], 1), threshold), right=False) - 1
    masked_ids = th.maximum(masked_ids, th.zeros_like(masked_ids))
    probs_threshold = probs_sorted.gather(-1, masked_ids)
    probs_mask = probs <= probs_threshold
    return probs_mask


def filter_cdf_prob(probs: th.Tensor, threshold: float):
    """
    Filter out all probabilities that are below the threshold.
    """
    probs_mask = filter_prob(probs, threshold)
    out = probs.clone()
    out[probs_mask] = 1e-8  # mask out all probabilities that are below the threshold
    return out


def filter_cdf(logits: th.Tensor, threshold: float):
    probs = logits.softmax(dim=-1)
    probs_mask = filter_prob(probs, threshold)
    out = logits.clone()
    out[probs_mask] = -1000
    return out


def round_to_multiple(x, N):
    pad = (N - x % N) % N
    return x + pad


def sort_2d(x):
    """It sorts descending order"""
    n_elem = x.shape[-1]
    x = x.view(-1)
    x_sort, ids = th.sort(x, descending=True)
    n_rows = ids // n_elem
    n_cols = ids % n_elem
    return x_sort, n_rows, n_cols


def to_th(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    return th.tensor(x, dtype=dtype, device=device)


def make_prefix(obs, transition_dim, device="cuda"):
    obs_discrete = to_th(obs, dtype=th.float32, device=device)
    pad_dims = to_th(np.zeros(transition_dim - len(obs)), dtype=th.float32, device=device)
    if obs_discrete.ndim == 1:
        obs_discrete = obs_discrete.reshape(1, 1, -1)
        pad_dims = pad_dims.reshape(1, 1, -1)
    transition = th.cat([obs_discrete, pad_dims], axis=-1)
    prefix = transition
    return prefix


def extract_actions(x, observation_dim, action_dim, t=None):
    actions = x[:, observation_dim:observation_dim + action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions


def extract_actions_continuous(x, observation_dim, action_dim, t=None):
    assert x.shape[0] == 1
    actions = x[0, :, observation_dim:observation_dim + action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions


def update_context(observation, action, reward, device):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])
    context = []

    transition_discrete = to_th(transition, dtype=th.float32, device=device)
    transition_discrete = transition_discrete.reshape(1, 1, -1)

    ## add new transition to context
    context.append(transition_discrete)
    return context


class Progress:
    def __init__(self, total, name='Progress', ncol=3, max_length=30, indent=8, line_width=100, speed_update_freq=100):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = '\033[F'
        self._clear_line = ' ' * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = '#' * self._pbar_size
        self._incomplete_pbar = ' ' * self._pbar_size

        self.lines = ['']
        self.fraction = '{} / {}'.format(0, self.total)

        self.resume()

    def update(self, description, n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print('\n', end='')
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=None):
        if params is None:
            params = []

        if type(params) == dict:
            params = sorted([
                (key, val)
                for key, val in params.items()
            ])

        # Position
        self._clear()

        # Percent
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        # Speed
        speed = self._format_speed(self._step)

        # Params
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = '{} | {}{}'.format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end='')
        print(empty)
        print(position, end='')

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
            fraction = '{} / {}'.format(n, total)
            string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
        else:
            fraction = '{}'.format(n)
            string = '{} iterations'.format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = '{:.1f} Hz'.format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, '')
        padding = '\n' + ' ' * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = ' | '.join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param, str_length=8):
        k, v = param
        k = k.rjust(str_length)
        if isinstance(v, float) or hasattr(v, 'item'):
            string = '{}: {:12.4f}'
        else:
            string = '{}: {}'
            v = str(v).rjust(12)
        return string.format(k, v)[:self.max_length]

    def stamp(self):
        if self.lines != ['']:
            params = ' | '.join(self.lines)
            string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
            string = re.sub(r'\s+', ' ', string)
            self._clear()
            print(string, end='\n')
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None


if __name__ == '__main__':
    silent = Silent()
    silent.update()
    silent.stamp()

    num_steps = 1000
    progress = Progress(num_steps)
    for i in range(num_steps):
        # progress.update(description='test')
        params = [
            ['A', '{:06d}'.format(i)],
            ['B', '{:06d}'.format(i)],
            ['C', '{:06d}'.format(i)],
            ['D', '{:06d}'.format(i)],
            ['E', '{:06d}'.format(i)],
            ['F', '{:06d}'.format(i)],
            ['G', '{:06d}'.format(i)],
            ['H', '{:06d}'.format(i)],
        ]
        progress.update(params)
        # progress.set_description(params)
        time.sleep(0.01)
    progress.close()
