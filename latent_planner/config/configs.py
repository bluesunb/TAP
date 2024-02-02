import os
import torch as th
import pickle
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union, Any
from datetime import datetime


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


@dataclass
class ConfigBase:
    save_dir: str

    # def __post_init__(self):
    #     self.save()

    def __repr__(self) -> str:
        string = f'\nConfig: {self.__class__.__name__}\n'
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            string += f'    {key}: {val}\n'
        return string

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        return iter(self.get_dict())

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        return len(self.get_dict())

    def get_dict(self) -> Dict[str, Any]:
        d = self.__dict__
        d.pop('save_dir')
        return d

    def save(self):
        if self.save_dir is not None:
            save_dir = os.path.join(*self.save_dir) if type(self.save_dir) is tuple else self.save_dir
            os.makedirs('/'.join(save_dir.split('/')[:-1]), exist_ok=True)
            pickle.dump(self.__dict__, open(save_dir, 'wb'))
            open(save_dir.replace('.pkl', '.txt'), 'w').write(str(self))
            print(f'Saved config to {save_dir}')

        return self


@dataclass
class DefaultConfig:
    model = "VQTransformer"
    tag = "experiment"
    state_conditional = True
    vocab_size = 100
    discount = 0.99
    n_layers = 4
    n_heads = 4

    ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
    n_epochs_ref = 50
    n_saves = 3
    # log_dir = '~/PycharmProjects/OfflineRL/TAP/scripts/logs'
    log_dir = './logs'
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    n_tokens = 360
    latent_step = 3
    emb_dim = 128  # n_embd
    traj_emb_dim = 512
    batch_size = 512
    learning_rate = 8e-4
    lr_decay = False
    seed = 42

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    step = 1
    subsampled_seq_len = 25
    termination_penalty = -100
    exp_name = f'vqvae'
    now = get_now_str()

    position_weight = 1
    action_weight = 5
    reward_weight = 1
    value_weight = 1

    first_action_weight = 0
    sum_reward_weight = 0
    last_value_weight = 0
    suffix = ''

    normalize_state = True
    normalize_reward = True
    max_path_length = 1001
    bottleneck = "pooling"
    masking = "uniform"
    disable_goal = False
    residual = True
    ma_update = True

    env_name: str = "halfcheetah-medium-expert-v0"
    dataset = None
    _savepath = None
    task_type = 'locomotion'
    obs_shape: List[int] = field(default_factory=lambda: [-1])

    @property
    def save_dir(self):
        return os.path.join(os.path.expanduser(self.log_dir), self.env_name, self.exp_name)

    @save_dir.setter
    def save_dir(self, value):
        self._savepath = value


@dataclass
class DatasetConfig(ConfigBase):
    save_dir: str
    env_name: Any
    dataset: str
    termination_penalty: int
    seq_len: int
    step: int
    discount: float
    disable_goal: bool
    normalize_sa: bool
    normalize_reward: bool
    max_path_length: int
    min_path_length: int
    device: str
    train_portion: float = 1.0

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class TransformerConfig(ConfigBase):
    save_dir: str
    n_layers: int

    emb_dim: int
    n_heads: int
    block_size: int
    attn_dropout_rate: float
    resid_dropout_rate: float
    emb_dropout_rate: float

    observation_dim: int
    action_dim: int
    transition_dim: int  # (obs_dim + act_dim + rew_dim + val_dim)

    traj_emb_dim: int  # trajectory_embd

    # vocab_size: int
    # Weights
    action_weight: float
    reward_weight: float
    value_weight: float
    pos_weight: float
    first_action_weight: float
    sum_reward_weight: float
    last_value_weight: float

    vocab_size: int  # N
    n_tokens: int  # K
    latent_step: int
    state_conditional: bool
    masking: str = "none"
    residual: bool = True
    bottleneck: bool = "pooling"
    ma_update: bool = False

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def traj_input_dim(self) -> int:
        return self.block_size - self.transition_dim

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)


@dataclass
class TrainerConfig(ConfigBase):
    save_dir: str
    batch_size: int
    learning_rate: float
    betas: Tuple[float, float]
    grad_norm_clip: float
    weight_decay: float
    lr_decay: bool
    warmup_tokens: int
    kl_warmup_tokens: int
    final_tokens: int
    num_workers: int
    device: str

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class PlannerConfig(ConfigBase):
    task_type: str

    discrete: bool = False
    gpt_epoch: str = 'latest'
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    renderer: str = 'human'
    suffix: str = '0'

    plan_freq: int = 1
    horizon: int = 15
    rounds: int = 2
    nb_samples: int = 4096
    beam_width: int = 64
    n_expend: int = 4
    prob_threshold: float = 0.05
    prob_weight: float = 500.0

    vis_freq: int = 200
    uniform: bool = False

    test_planner: str = "beam_prior"

    def __repr__(self) -> str:
        return super().__repr__()


def load_configs(path: str) -> Tuple["DatasetConfig", "TransformerConfig", "TrainerConfig"]:
    dataset_config = pickle.load(open(os.path.join(path, 'data_config.pkl'), 'rb'))
    model_config = pickle.load(open(os.path.join(path, 'model_config.pkl'), 'rb'))
    trainer_config = pickle.load(open(os.path.join(path, 'trainer_config.pkl'), 'rb'))
    if isinstance(dataset_config, dict):
        return (DatasetConfig(**dataset_config),
                TransformerConfig(**model_config),
                TrainerConfig(**trainer_config))

    return dataset_config, model_config, trainer_config


def load_config(path: str, cls):
    config = pickle.load(open(path, 'rb'))
    if isinstance(config, dict):
        return cls(**config)
    return config


def get_recent(path: str):
    dirs = os.listdir(path)
    return os.path.join(path, max(dirs))

def get_recent_model_name(path: str, prefix: str):
    dirs = os.listdir(path)
    dirs = [d for d in dirs if d.startswith(prefix) and d.endswith('.pt')]
    return max(dirs, key=lambda x: int(x.split('_')[1].split('.')[0]))
