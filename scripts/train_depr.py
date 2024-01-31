import os
import numpy as np
import torch as th

# from latent_planner.models.training import VQTrainer
# from latent_planner.utils.rendering import get_preprocess_fn, load_environment
from latent_planner.train_utils.training import VQTrainer
import latent_planner.datasets as datasets
from latent_planner.models.autoencoders import VQContinuousVAE

from latent_planner.config import DefaultConfig, DatasetConfig, TransformerConfig, TrainerConfig


config = DefaultConfig()

# ============= Dataset =============

# env_name = "halfcheetah-medium-expert-v0"
env_name = "maze2d-medium-v1"
env = datasets.load_environment(env_name)
config.dataset = env_name

seq_len = config.subsampled_seq_len * config.step
config.log_dir = os.path.expanduser(config.log_dir)
config.save_dir = os.path.expanduser(config.save_dir)
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

dataset_class = datasets.SequenceDataset
dataset_config = DatasetConfig(
    save_dir=(config.save_dir, 'data_config.pkl'),
    env=config.dataset,
    termination_penalty=config.termination_penalty,
    seq_len=seq_len,
    step=config.step,
    discount=config.discount,
    disable_goal=config.disable_goal,
    normalize_sa=config.normalize_state,
    normalize_reward=config.normalize_reward,
    max_path_length=int(config.max_path_length),
    device=config.device
)

dataset = dataset_class(dataset_config)
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = act_dim + 3
transition_dim += obs_dim if config.task_type == "locomotion" else 128


# ============= Model =============

# Total dimension of the input to the transformer
block_size = config.subsampled_seq_len * transition_dim

print(f'Dataset size: {len(dataset)} | '
      f'Joined dim: {transition_dim} '
      f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}')


model_config = TransformerConfig(
    save_dir=os.path.join(config.save_dir, 'model_config.pkl'),
    vocab_size=config.vocab_size,
    block_size=block_size,

    n_tokens=config.n_tokens,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    emb_dim=config.emb_dim * config.n_heads,

    observation_dim=obs_dim,
    action_dim=act_dim,
    transition_dim=transition_dim,

    traj_emb_dim=config.traj_emb_dim,
    latent_step=config.latent_step,
    ma_update=config.ma_update,
    residual=config.residual,
    # obs_shape=config.obs_shape,
    # model=config.model,

    action_weight=config.action_weight,
    reward_weight=config.reward_weight,
    value_weight=config.value_weight,
    pos_weight=config.position_weight,
    first_action_weight=config.first_action_weight,
    sum_reward_weight=config.sum_reward_weight,
    last_value_weight=config.last_value_weight,

    emb_dropout_rate=config.embd_pdrop,
    attn_dropout_rate=config.attn_pdrop,
    resid_dropout_rate=config.resid_pdrop,

    bottleneck=config.bottleneck,
    masking=config.masking,
    state_conditional=config.state_conditional
)

model = VQContinuousVAE(model_config)
model.to(config.device)
if config.normalize_state:
    padding_vector = dataset.normalize_joined_single(np.zeros(model_config.transition_dim - 1))
    model.padding_vector = th.from_numpy(padding_vector).to(model.padding_vector)


# ============= Training =============

warmup_tokens = len(dataset) * block_size    # number of tokens seen per epoch
final_tokens = 20 * warmup_tokens            # total number of tokens in dataset

trainer_config = TrainerConfig(
    save_dir=os.path.join(config.save_dir, 'trainer_config.pkl'),
    batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1,
    lr_decay=config.lr_decay,
    warmup_tokens=warmup_tokens,
    kl_warmup_tokens=warmup_tokens * 10,
    final_tokens=final_tokens,
    num_workers=8,
    device=config.device
)

trainer = VQTrainer(trainer_config)


# ============= Main loop =============

n_epochs = int(1e6 / len(dataset) * config.n_epochs_ref)
# n_epochs = 1
save_freq = int(n_epochs // config.n_saves)

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch}/{n_epochs} | {config.dataset} | {config.exp_name}")
    
    trainer.train(model, dataset, log_freq=50)

    save_epoch = (epoch + 1) // save_freq * save_freq
    state_path = os.path.join(config.save_dir, f"state_{save_epoch}.pt")
    print(f"Saving model to {state_path}")

    state = model.state_dict()
    th.save(state, state_path)
