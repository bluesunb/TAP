import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any
from latent_planner.config import TransformerConfig
from latent_planner.models.transformers import AttentionBlock, AsymAttentionBlock
from latent_planner.models.vqvae import (VQEmbedding,
                                         VQEmbeddingEMA,
                                         VectorQuantization)
from latent_planner.models.utils import configure_optimizer, init_weights


class Encoder(nn.Module):
    def __init__(self, layer_arch: List[int], latent_dim: int, condition_dim: int):
        """
        Encoder module for the VAE

        Args:
            layer_arch (List[int]): List of output dimensions for each linear layer in MLP.
            latent_dim (int): Dimension of the latent space.
            condition_dim (int): Dimension of the condition space.

        Notes:
            - inputs: (bs, seq_len, traj_input_dim)
            - outputs:
                1. (bs, seq_len, latent_dim)
                2. (bs, seq_len, latent_dim)
        """
        super().__init__()
        layer_arch[0] += condition_dim  # concatenate condition to input
        self.mlp = self._construct_mlp(layer_arch)
        self.linear_means = nn.Linear(layer_arch[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_arch[-1], latent_dim)

    @staticmethod
    def _construct_mlp(layer_arch: List[int]) -> nn.Module:
        """
        Construct a sequential MLP from the given layer architecture.

        Args:
            layer_arch (List[int]): List of output dimensions for each linear layer in MLP.

        Returns:
            nn.Module: Sequential MLP following the given layer architecture.
        """
        layers = []
        for dim_in, dim_out in zip(layer_arch[:-1], layer_arch[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        It predicts the mean and log variance of the latent distribution.
        Args:
            x: (bs, seq_len, traj_input_dim): Input sequence of trajectory.

        Returns:
            (bs, seq_len, latent_dim): Mean of the latent distribution.
            (bs, seq_len, latent_dim): Log variance of the latent distribution.
        """
        x = self.mlp(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_arch: List[int], latent_dim: int, condition_dim: int):
        """
        Decoder module for the VAE

        Args:
            layer_arch (List[int]): List of output dimensions for each linear layer in MLP.
            latent_dim (int): Dimension of the latent space.
            condition_dim (int): Dimension of the condition space.

        Notes:
            - inputs: (bs, seq_len, latent_dim + condition_dim)
            - outputs: (bs, seq_len, emb_dim)
        """
        super().__init__()

        layer_arch = [latent_dim + condition_dim] + layer_arch
        self.mlp = self._construct_mlp(layer_arch)

    @staticmethod
    def _construct_mlp(layer_arch: List[int]) -> nn.Module:
        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(layer_arch[:-1], layer_arch[1:])):
            layers.append(nn.Linear(dim_in, dim_out))
            if i != len(layer_arch) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, z: th.Tensor):
        """
        Args:
            z: (bs, seq_len, latent_dim + condition_dim): Latent sequence.

        Returns:
            (bs, seq_len, emb_dim): Output sequence of embeddings.
        """
        return self.mlp(z)


class MLPModel(nn.Module):
    """
    Autoencoder model with MLP encoder and decoder.

    Attributes:
        condition_dim (int): Dimension of the condition space. In this case, condition is the desire state.
        traj_input_dim (int): Dimension of the input trajectory sequence.
        encoder (Encoder): MLP encoder.
        decoder (Decoder): MLP decoder.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.condition_dim = config.observation_dim
        # self.traj_input_dim = config.block_size - config.transition_dim
        self.traj_input_dim = config.traj_input_dim

        encoder_arch = [self.traj_input_dim, 512, 256]
        decoder_arch = encoder_arch[::-1]

        self.encoder = Encoder(encoder_arch, latent_dim=config.traj_emb_dim, condition_dim=0)
        self.decoder = Decoder(decoder_arch, latent_dim=config.traj_emb_dim, condition_dim=self.condition_dim)

    def encode(self, x: th.Tensor) -> th.Tensor:
        """
        Encode the raw input sequence into a latent sequence.
        Input sequence is a generally a trajectory of observations.

        Args:
            x: (bs, seq_len, transition_dim): Input sequence of observations.

        Returns:
            (bs, seq_len, latent_dim): Latent sequence.
        """
        inputs = th.flatten(x, start_dim=1)
        latents = self.encoder(inputs)
        return latents

    def decode(self, z: th.Tensor, condition: th.Tensor) -> th.Tensor:
        """
        Decode (reconstruct) the latent sequence into a sequence of embeddings.

        Args:
            z: (bs, seq_len, latent_dim): Latent sequence.
            condition: (bs, seq_len, observation_dim): State sequence which is used as condition.

        Returns:
            (bs, seq_len, emb_dim): Output sequence of embeddings.
        """
        condition_state = th.flatten(condition, start_dim=1)
        mixed_latent = th.cat([condition_state, z], dim=-1)
        recon = self.decoder(mixed_latent)
        return recon


class SymbolwiseTransformer(nn.Module):
    """
    Encode the sequence of transitions (s,a,r,v) into a latent distribution by separate each transition token
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.traj_emb_dim
        self.condition_dim = config.observation_dim
        # self.traj_input_dim = config.block_size - config.transition_dim
        self.traj_input_dim = config.traj_input_dim
        self.traj_length = 4 * (config.block_size // config.transition_dim - 1)

        self.encoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])

        # Token embeddings
        self.pos_emb = nn.Parameter(th.zeros(1, self.traj_length, config.emb_dim))
        self.state_emb = nn.Linear(config.observation_dim, config.emb_dim)
        self.action_emb = nn.Linear(config.action_dim, config.emb_dim)
        self.reward_emb = nn.Linear(1, config.emb_dim)
        self.value_emb = nn.Linear(1, config.emb_dim)

        # Token prediction heads
        self.pred_state = nn.Linear(config.emb_dim, config.observation_dim)
        self.pred_action = nn.Linear(config.emb_dim, config.action_dim)
        self.pred_reward = nn.Linear(config.emb_dim, 1)
        self.pred_value = nn.Linear(config.emb_dim, 1)

        # Latent encoder
        self.linear_means = nn.Linear(config.emb_dim, self.latent_dim)
        self.linear_log_var = nn.Linear(config.emb_dim, self.latent_dim)
        self.mixed_latent_emb = nn.Linear(self.latent_dim + config.observation_dim, config.emb_dim)

        self.ln = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.emb_dropout_rate)

    def encode(self, traj_sequence: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Encode the conditioned inputs into a latent mean and log variance.

        Args:
            traj_sequence: (bs, seq_len, joined_dim): Sequence of transitions,
                            where `joined_dim` = obs_dim + act_dim + 2 (reward & value)

        Returns:
            (bs, latent_dim): Mean of the trajectory latent distribution.
            (bs, latent_dim): Log variance of the trajectory latent distribution.
        """
        bs, seq_len, joined_dim = traj_sequence.shape
        assert seq_len <= self.config.block_size, \
            f"Sequence length ({seq_len}) must be less than block size ({self.config.block_size})"

        split = np.cumsum([self.config.observation_dim, self.config.action_dim, 1, 1])
        states, actions, rewards, values = th.split(traj_sequence, split, dim=-1)

        # embeddings: (bs, seq_len, emb_dim)
        state_embeddings = self.state_emb(states)
        action_embeddings = self.action_emb(actions)
        reward_embeddings = self.reward_emb(rewards)
        value_embeddings = self.value_emb(values)

        # sequentially concatenate all embeddings: [(s1, a1, r1, v1), (s2, a2, r2, v2), ...] -> (bs, 4 * seq_len, emb_dim)
        token_embeddings = th.stack([state_embeddings, action_embeddings, reward_embeddings, value_embeddings], dim=1)
        token_embeddings = token_embeddings.permute(0, 2, 1, 3)  # (bs, seq_len, 4, emb_dim)
        token_embeddings = token_embeddings.reshape(bs, 4 * seq_len, self.config.emb_dim)  # (bs, 4 * seq_len, emb_dim)

        pos_embeddings = self.pos_emb[:, :token_embeddings.size(1), :]  # (1, 4 * seq_len, emb_dim)
        traj_embeddings = self.dropout(pos_embeddings + token_embeddings)
        traj_embeddings = self.encoder(traj_embeddings)  # (bs, 4 * seq_len, emb_dim)

        traj_features = traj_embeddings.max(dim=1).values  # (bs, emb_dim)
        means = self.linear_means(traj_features)  # (bs, latent_dim)
        log_vars = self.linear_log_var(traj_features)  # (bs, latent_dim)
        return means, log_vars

    def decode(self, z: th.Tensor, condition: th.Tensor) -> th.Tensor:
        """
        Predict (a_t, r_t, v_t, s_(t+1)) from the latent of (c | (s,a,r,v)_{:t+1})
        Args:
            z: (bs, latent_dim): latent of previous transitions
            condition: (bs, observation_dim): state as condition

        Returns:
            (bs, traj_dim): predicted (a_t, r_t, v_t, s_(t+1)) sequence,
                            where traj_dim = obs_dim + act_dim + reward(1) + value(1)
        """
        condition_state = th.flatten(condition, start_dim=1)  # (bs, obs_dim)
        mixed_latent = th.cat([condition_state, z], dim=-1)  # (bs, obs_dim + latent_dim)
        mixed_latent = self.mixed_latent_emb(mixed_latent)  # (bs, emb_dim)
        mixed_latent = mixed_latent[:, None, :] + self.pos_emb[:, :]  # (bs, 4 * seq_len, emb_dim)

        traj_decoded = self.decoder(mixed_latent)  # (bs, 4 * seq_len, emb_dim)
        traj_decoded = self.ln(traj_decoded)

        # traj_embeddings: (s, a, r, v)
        traj_decoded = rearrange(traj_decoded, "b (t s) e -> b s t e", s=4)  # (bs, 4, seq_len, emb_dim)

        state_pred = self.pred_state(traj_decoded[:, 1])  # s' ~ P(.|s, a) -> next state
        action_pred = self.pred_action(traj_decoded[:, 0])  # a ~ P(.|s) -> current action
        reward_pred = self.pred_reward(traj_decoded[:, 1])  # r ~ r(.|s, a) -> current reward
        value_pred = self.pred_value(traj_decoded[:, 1])  # v ~ v(.|s, a) -> current value

        traj_pred = th.cat([state_pred, action_pred, reward_pred, value_pred], dim=-1)  # (bs, seq_len, traj_dim)
        return traj_pred


class StepwiseTransformer(nn.Module):
    """
    Encode the sequence of transitions (s,a,r,v) into a latent distribution by input the entire sequence at once
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.latent_dim = config.traj_emb_dim
        self.condition_dim = config.observation_dim
        # self.traj_input_dim = config.block_size - config.transition_dim
        self.traj_input_dim = config.traj_input_dim
        self.traj_length = config.block_size // config.transition_dim - 1

        self.encoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])

        self.pos_emb = nn.Parameter(th.zeros(1, self.traj_length, config.emb_dim))
        self.embed = nn.Linear(config.transition_dim, config.emb_dim)
        self.pred = nn.Linear(config.emb_dim, config.transition_dim)

        self.linear_means = nn.Linear(config.emb_dim, self.latent_dim)
        self.linear_log_var = nn.Linear(config.emb_dim, self.latent_dim)
        self.mixed_latent_emb = nn.Linear(self.latent_dim + config.observation_dim, config.emb_dim)

        self.ln = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.emb_dropout_rate)

    def encode(self, traj_sequence: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            traj_sequence: (bs, seq_len, joined_dim = transition_dim): Sequence of transitions,
                            where `joined_dim` = obs_dim + act_dim + reward(1) + value(1)

        Returns:
            (bs, latent_dim): Mean of the trajectory latent distribution.
            (bs, latent_dim): Log variance of the trajectory latent distribution.
        """
        seq_len = traj_sequence.size(1)
        assert seq_len <= self.config.block_size, \
            f"Sequence length ({seq_len}) must be less than block size ({self.config.block_size})"

        token_embeddings = self.embed(traj_sequence)  # (bs, seq_len, emb_dim)
        pos_embeddings = self.pos_emb[:, :token_embeddings.size(1), :]  # (bs, seq_len, emb_dim)

        traj_embeddings = self.dropout(pos_embeddings + token_embeddings)
        traj_embeddings = self.encoder(traj_embeddings)  # (bs, seq_len, emb_dim)

        traj_features = traj_embeddings.max(dim=1).values
        means = self.linear_means(traj_features)  # (bs, latent_dim)
        log_vars = self.linear_log_var(traj_features)  # (bs, latent_dim)
        return means, log_vars

    def decode(self, z: th.Tensor, condition: th.Tensor) -> th.Tensor:
        """
        Decode the latent sequence into a sequence of transitions.

        Args:
            z: (bs, latent_dim): Latent sequence.
            condition: (bs, obs_dim): State as condition.

        Returns:
            (bs, seq_len, transition_dim): Output sequence of transitions.
        """
        condition_state = th.flatten(condition, start_dim=1)  # (bs, obs_dim)
        mixed_latent = th.cat([condition_state, z], dim=-1)  # (bs, obs_dim + latent_dim)
        mixed_latent = self.mixed_latent_emb(mixed_latent)  # (bs, emb_dim)
        mixed_latent = mixed_latent[:, None, :] + self.pos_emb[:, :]  # (bs, seq_len, emb_dim)

        traj_decoded = self.decoder(mixed_latent)  # (bs, seq_len, emb_dim)
        traj_decoded = self.ln(traj_decoded)

        traj_pred = self.pred(traj_decoded)  # (bs, seq_len, transition_dim)
        traj_pred[:, :, -1] = th.sigmoid(traj_pred[:, :, -1])  # sigmoid for value to be in [0, 1]
        return traj_pred


class VQStepwiseTransformer(nn.Module):
    """
    Take the sequence of transitions as input and reconstruct it by Transformer (encoder, decode).
    VQ-VAE is used to encode the trajectory embeddings into a latent sequence that taken by the decoder.

    Notes:
        - inputs:
            1. traj_sequence: (bs, seq_len, joined_dim = transition_dim): Sequence of transitions,
                                where `joined_dim` = obs_dim + act_dim + reward(1) + value(1)
            2. condition: (bs, obs_dim): State as condition.

        - outputs:
            1. traj_pred: (bs, seq_len, transition_dim): Output sequence of transitions.
            2. z_q_x_bar: (bs, seq_len, emb_dim): Latent sequence.
            3. z_e_x:
    """

    def __init__(self, config: TransformerConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.config.observation_dim = feature_dim

        self.n_tokens = config.n_tokens
        self.latent_dim = config.traj_emb_dim
        self.condition_dim = config.observation_dim
        self.traj_input_dim = config.traj_input_dim
        self.traj_length = config.block_size // config.transition_dim - 1

        self.masking = config.get("masking", "none")
        self.residual = config.get('residual', True)
        self.bottleneck = config.get("bottleneck", "pooling")
        self.ma_update = config.get("ma_update", True)

        self.encoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])

        self.pos_emb = nn.Parameter(th.zeros(1, self.traj_length, config.emb_dim))
        self.embed = nn.Linear(config.transition_dim, config.emb_dim)
        self.cast_embed = nn.Linear(config.emb_dim, self.latent_dim)

        self.mixed_latent_emb = nn.Linear(self.latent_dim + config.observation_dim, config.emb_dim)
        self.pred = nn.Linear(config.emb_dim, config.transition_dim)

        self.ln = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.emb_dropout_rate)

        if self.ma_update:
            self.codebook = VQEmbeddingEMA(self.n_tokens, config.traj_emb_dim, decay=0.99)
        else:
            self.codebook = VQEmbedding(self.n_tokens, config.traj_emb_dim)

        if self.bottleneck == "pooling":
            self.latent_pooling = nn.MaxPool1d(config.latent_step, stride=config.latent_step)
        elif self.bottleneck == "attention":
            self.out_tokens = self.traj_length // config.latent_step
            self.latent_pooling = AsymAttentionBlock(config, out_tokens=self.out_tokens)
            self.expand = AsymAttentionBlock(config, out_tokens=self.traj_length)
        else:
            raise ValueError(f"Invalid bottleneck type: {self.bottleneck}")

    def encode(self, traj_sequence: th.Tensor) -> th.Tensor:
        """
        Encode the sequence of transitions into a latent sequence.

        Args:
            traj_sequence: (bs, seq_len, joined_dim): Sequence of transitions,
                            where `joined_dim` = obs_dim + act_dim + reward(1) + value(1) = transition_dim - 1
        Returns:
            (bs, red_len, latent_dim): Latent sequence.
        """
        traj_sequence = traj_sequence.to(th.float32)
        seq_len = traj_sequence.size(1)
        assert seq_len <= self.config.block_size, \
            f"Sequence length ({seq_len}) must be less than block size ({self.config.block_size})"

        token_embeddings = self.embed(traj_sequence)  # (bs, seq_len, emb_dim)
        pos_embeddings = self.pos_emb[:, :token_embeddings.size(1), :]  # (bs, seq_len, emb_dim)

        traj_embeddings = self.dropout(pos_embeddings + token_embeddings)
        traj_embeddings = self.encoder(traj_embeddings)  # (bs, seq_len, emb_dim)

        if self.bottleneck == "pooling":
            # traj_embeddings : (bs, red_len = seq_len // latent_step, latent_dim)
            traj_embeddings = self.latent_pooling(traj_embeddings.transpose(1, 2)).transpose(1, 2)
        elif self.bottleneck == "attention":
            # traj_embeddings : (bs, red_len = traj_length // latent_step, latent_dim)
            traj_embeddings = self.latent_pooling(traj_embeddings)

        traj_embeddings = self.cast_embed(traj_embeddings)  # (bs, red_len, latent_dim)
        return traj_embeddings

    def decode(self, z: th.Tensor, condition: th.Tensor) -> th.Tensor:
        """
        Decode the latent sequence into a sequence of transitions.

        Args:
            z: (bs, red_len, latent_dim): Latent sequence.
            condition: (bs, obs_dim): State as condition.

        Returns:
            (bs, seq_len, transition_dim): Output sequence of transitions.
        """
        bs, reduced_len = z.shape[:2]  # red_len = seq_len // latent_step
        condition_state = condition.view(bs, 1, -1).repeat(1, reduced_len, 1)  # (bs, red_len, obs_dim)
        if not self.config.state_conditional:
            condition_state = th.zeros_like(condition_state)

        mixed_latent = th.cat([condition_state, z], dim=-1)  # (bs, red_len, obs_dim + latent_dim)
        mixed_latent = self.mixed_latent_emb(mixed_latent)  # (bs, red_len, emb_dim)

        if self.bottleneck == "pooling":
            mixed_latent = th.repeat_interleave(mixed_latent, self.config.latent_step, dim=1)  # (bs, seq_len, emb_dim)
        elif self.bottleneck == "attention":
            mixed_latent = self.expand(mixed_latent)
        mixed_latent = mixed_latent + self.pos_emb[:, :mixed_latent.size(1), :]  # (bs, seq_len, emb_dim)

        traj_decoded = self.decoder(mixed_latent)  # (bs, seq_len, emb_dim)
        traj_decoded = self.ln(traj_decoded)

        traj_pred = self.pred(traj_decoded)  # (bs, seq_len, transition_dim)
        traj_pred[:, :, -1] = th.sigmoid(traj_pred[:, :, -1])  # sigmoid for value to be in [0, 1]
        traj_pred[:, :, :self.config.observation_dim] += th.reshape(condition, (bs, 1, -1))  # residual connection
        return traj_pred

    def forward(self, traj_sequence: th.Tensor, condition: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode trajectory into a latent sequence by VQ-VAE then compute reconstructed trajectory.
        Args:
            traj_sequence: (bs, seq_len, joined_dim): Sequence of transitions,
                            where `joined_dim` = obs_dim + act_dim + reward(1) + value(1) + done(1) = transition_dim
            condition: (bs, obs_dim): State as condition.

        Returns:
            (bs, seq_len, transition_dim): Reconstructed trajectory.
            (bs, red_len, latent_dim): Latent sequence (z_q_x).
            (bs, red_len, latent_dim): trajectory embeddings (z_e_x).
        """
        traj_embeddings = self.encode(traj_sequence)  # (bs, red_len, latent_dim)
        latents_prev, latents = self.codebook.straight_through(traj_embeddings)  # (bs, red_len, latent_dim)

        bs, reduced_len = latents_prev.shape[:2]
        if self.bottleneck == "attention":
            # if bottleneck method is attention, we mask it uniformly to prevent the model from learning the position
            if self.masking == "uniform":
                mask = th.ones(bs, reduced_len, 1).to(latents_prev.device)
                # uniformly select the index to start masking
                mask_idx = np.random.randint(0, reduced_len, size=(bs, 1))
                for n, idx in enumerate(mask_idx):
                    mask[n, -idx:, 0] = 0
                # mask out
                latents_prev = latents_prev * mask
                latents = latents * mask

            elif self.masking != "none":
                raise ValueError(f"Invalid masking type: {self.masking}")

        traj_recon = self.decode(latents_prev, condition)  # (bs, seq_len, transition_dim)
        return traj_recon, latents, traj_embeddings


class VQContinuousVAE(nn.Module):
    """
    VQ-VAE with Transformer encoder and decoder.
    It takes the reduced transition sequence (red_dim) as input and embed it to VQ-VAE then reconstruct it by Transformer.

    - inputs:
        1. traj_sequence: (bs, seq_len, red_dim): Sequence of transitions without terminal token.
        2. targets: (bs, seq_len, red_dim): True next step sequence of transitions.
        3. mask: (bs, seq_len): Masking for the reconstruction loss.
        4. terminals: (bs, seq_len, 1): Terminal mask.

    - outputs:
        1. traj_recon: (bs, seq_len, transition_dim): Reconstructed trajectory.
        2. recon_loss: (bs, ): Reconstruction loss.
            = mse(traj_recon, targets) + first_action_loss + sum_reward_loss + last_value_loss + term_pred_loss
        3. vq_loss: (bs, ): VQ loss.
        4. commit_loss: (bs, ): Commitment loss.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model = VQStepwiseTransformer(config, config.observation_dim)

        # self.vocab_size = config.vocab_size
        self.stop_token = config.n_tokens * config.transition_dim
        self.traj_length = config.block_size // config.transition_dim - 1
        assert self.traj_length % config.latent_step == 0, \
            f"Trajectory length ({self.traj_length}) must be divisible by latent step ({config.latent_step})"
        self.masking = config.get("masking", "none")

        self.padding_vector = th.zeros(config.transition_dim - 1)  # transition_dim = obs + act + reward + value + term
        self.apply(init_weights)

    def configure_optimizers(self, train_config: Dict[str, Any]) -> th.optim.Optimizer:
        return configure_optimizer(model=self,
                                   learning_rate=train_config["learning_rate"],
                                   weight_decay=train_config["weight_decay"],
                                   betas=train_config["betas"],
                                   submodel=self.model)

    @th.no_grad()
    def encode(self, traj_sequence: th.Tensor, terminals: th.Tensor) -> th.Tensor:
        """
        Encode the sequence of transitions into a latent sequence then quantize the latent sequence.

        Args:
            traj_sequence: (bs, seq_len, transition_dim - 1): Sequence of transitions,
                            where `joined_dim` = obs_dim + act_dim + reward(1) + value(1)
            terminals: (bs, seq_len, 1): Terminal mask.

        Returns:
            (bs, red_len): Quantized latent indices.
        """
        bs, seq_len, joined_dim = traj_sequence.shape  # joined_dim = transition_dim - 1
        # pad = th.tensor(self.padding_vector, dtype=th.float32, device=traj_sequence.device)
        pad = self.padding_vector.clone().to(traj_sequence.device)
        pad = pad.repeat(bs, seq_len, 1)  # (bs, seq_len, transition_dim - 1)
        term_mask = th.clone(1 - terminals).repeat(1, 1, joined_dim)  # mark 0 for terminal transitions
        traj_sequence = traj_sequence * term_mask + (1 - term_mask) * pad

        traj_embeddings = self.model.encode(th.cat([traj_sequence, terminals], dim=-1))  # (bs, red_len, latent_dim)
        # indices: (bs, red_len)
        indices = VectorQuantization.apply(traj_embeddings,
                                           self.model.codebook.embedding if self.model.ma_update
                                           else self.model.codebook.embedding.weight)
        return indices

    def decode(self, z: th.Tensor, condition: th.Tensor) -> th.Tensor:
        return self.model.decode(z, condition)

    def decode_from_indices(self, indices: th.Tensor, condition: th.Tensor) -> th.Tensor:
        """
        Decode the quantized latent sequence into a sequence of transitions.

        Args:
            indices: (bs, red_len): Quantized latent sequence.
            condition:  (bs, obs_dim): State as condition.

        Returns:
            (bs, seq_len, transition_dim): Output sequence of transitions.
        """
        bs, reduced_len = indices.shape  # red_len = seq_len // latent_step
        latents = th.index_select(self.model.codebook.embedding if self.model.ma_update
                                  else self.model.codebook.embedding.weight,
                                  dim=0, index=indices.flatten())
        latents = latents.view(bs, reduced_len, -1)  # (bs, red_len, latent_dim)
        assert latents.shape[-1] == self.model.latent_dim

        if self.model.bottleneck == "attention":
            # latent : (bs, seq_len, latent_dim)
            latents = th.concat([latents, th.zeros(bs, self.model.out_tokens, latents.shape[-1]).to(latents)], dim=1)

        condition = condition.unsqueeze(1).repeat(latents.size(0), 1, 1)  # (bs, 1, obs_dim)
        return self.model.decode(latents, condition)  # (bs, seq_len, transition_dim)

    def forward(self,
                traj_sequence: th.Tensor,
                targets: Optional[th.Tensor] = None,
                mask: Optional[th.Tensor] = None,
                terminals: Optional[th.Tensor] = None):
        """
        Encode trajectory into a latent sequence by VQ-VAE then compute reconstructed trajectory.
        Also, calculate the reconstruction loss, VQ loss, and commitment loss for VQ-VAE

        Args:
            traj_sequence: (bs, seq_len, reduced_dim): Sequence of transitions,
                            where `reduced_dim` = obs_dim + act_dim + reward(1) + value(1) = transition_dim - 1
            targets: (bs, seq_len, reduced_dim) : True next step sequence of transitions.
            mask: (bs, seq_len): Masking for the reconstruction loss.
            terminals: (bs, seq_len, 1): Terminal mask.

        Returns:
            traj_recon: (bs, seq_len, transition_dim): Reconstructed trajectory.
            recon_loss: (bs, ): Reconstruction loss.
            vq_loss: (bs, ): VQ loss.
            commit_loss: (bs, ): Commitment loss.
        """

        traj_sequence = traj_sequence.to(th.float32)
        bs, seq_len, reduced_dim = traj_sequence.shape  # reduced_dim = transition_dim - 1
        obs_dim = self.config.observation_dim
        act_dim = self.config.action_dim

        # pad = th.tensor(self.padding_vector, dtype=th.float32, device=traj_sequence.device)
        pad = self.padding_vector.clone().to(traj_sequence.device)
        pad = pad.repeat(bs, seq_len, 1)  # (bs, seq_len, transition_dim - 1)

        term_mask = th.ones(1, 1, reduced_dim).to(traj_sequence.device)
        if terminals is not None:
            term_mask = th.clone(1 - terminals).repeat(1, 1, reduced_dim)
            traj_sequence = traj_sequence * term_mask + (1 - term_mask) * pad

        condition = traj_sequence[:, 0, :obs_dim]  # s0: (bs, obs_dim)
        # recon_traj: (bs, red_len, transition_dim)
        # latents, traj_embeddings: (bs, red_len, latent_dim)
        traj_recon, latents, traj_embeddings = self.model(th.cat([traj_sequence, terminals], dim=-1), condition)
        traj_pred, term_pred = th.split(traj_recon, [reduced_dim, 1], dim=-1)

        if targets is None:
            return traj_recon, None, None, None

        # Calculate losses
        device = traj_sequence.device
        weights = th.cat([
            th.full((2,), self.config.pos_weight),
            th.full((obs_dim - 2,), 1.0),
            th.full((act_dim,), self.config.action_weight),
            th.full((1,), self.config.reward_weight),
            th.full((1,), self.config.value_weight)
        ]).to(device)  # (transition_dim - 1, )

        traj_pred_loss = F.mse_loss(traj_pred, traj_sequence, reduction="none")  # (bs, red_len, transition_dim - 1)
        traj_pred_loss *= weights[None, None, :]

        first_action_loss = F.mse_loss(traj_pred[:, 0, obs_dim:obs_dim + act_dim],
                                       traj_sequence[:, 0, obs_dim:obs_dim + act_dim])
        first_action_loss *= self.config.first_action_weight  # (bs, )

        sum_reward_loss = F.mse_loss(traj_pred[:, :, -2].mean(dim=-1), traj_sequence[:, :, -2].mean(dim=-1))
        sum_reward_loss *= self.config.sum_reward_weight  # (bs, )

        last_value_loss = F.mse_loss(traj_pred[:, -1, -1], traj_sequence[:, -1, -1])
        last_value_loss *= self.config.last_value_weight  # (bs, )

        term_loss = F.binary_cross_entropy(term_pred, th.clip(terminals.float(), 0, 1))

        recon_loss = first_action_loss + sum_reward_loss + last_value_loss + term_loss
        recon_loss += (traj_pred_loss * mask * term_mask).mean()

        # vq_loss is 0 when using EMA which already updated the codebook by moving average
        vq_loss = th.zeros_like(recon_loss) if self.model.ma_update else F.mse_loss(latents, traj_embeddings.detach())
        commit_loss = F.mse_loss(traj_embeddings, latents.detach())

        return traj_recon, recon_loss, vq_loss, commit_loss


class TransformerPrior(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.n_tokens, config.emb_dim)
        self.pos_emb = nn.Parameter(th.zeros(1, config.block_size, config.emb_dim))
        self.state_emb = nn.Linear(config.observation_dim, config.emb_dim)

        self.transformer = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layers)])
        self.tok_pred = nn.Linear(config.emb_dim, config.n_tokens)

        self.ln = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.emb_dropout_rate)

        self.apply(init_weights)

    def configure_optimizers(self, train_config: Dict[str, Any]) -> th.optim.Optimizer:
        return configure_optimizer(model=self,
                                   learning_rate=train_config["learning_rate"],
                                   weight_decay=train_config["weight_decay"],
                                   betas=train_config["betas"])

    def forward(self, idx: th.Tensor, condition: th.Tensor, targets: Optional[th.Tensor] = None):
        """
        Embed the input sequence and predict the next (VQ)token in the sequence.

        Args:
            idx: (bs, seq_len): Input sequence of token indices.
            condition: (bs, obs_dim): State as condition.
            targets: (bs, seq_len): Target sequence of token indices.

        Returns:
            logits: (bs, seq_len + 1, n_tokens): Logits of the current and next token in the sequence.
            loss: (1, ): Cross entropy loss of the next token.
        """
        if idx is None:
            seq_len = 1
            tok_embeddings = th.zeros(1, 1, self.config.emb_dim).to(self.pos_emb)
        else:
            seq_len = idx.size(1)
            assert seq_len <= self.config.block_size, \
                f"Sequence length ({seq_len}) must be less than block size ({self.config.block_size})"

            tok_embeddings = self.tok_emb(idx)      # (bs, seq_len, emb_dim)
            tok_embeddings = F.pad(tok_embeddings, (0, 0, 1, 0, 0, 0), value=0)  # (bs, seq_len + 1, emb_dim)

        condition = condition.to(th.float32)
        pos_embeddings = self.pos_emb[:, :seq_len + 1]  # (1, seq_len + 1, emb_dim)
        state_embeddings = self.state_emb(condition).unsqueeze(1)  # (bs, 1, emb_dim)

        x = self.dropout(tok_embeddings + pos_embeddings + state_embeddings)
        x = self.transformer(x)     # (bs, seq_len + 1, emb_dim)
        x = self.ln(x)

        logits = self.tok_pred(x).view(x.shape[0], seq_len + 1, -1)   # (bs, seq_len + 1, n_tokens)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.n_tokens), targets.flatten(), reduction="mean")

        return logits, loss


class VQContinuousVAEEncWrap(nn.Module):
    def __init__(self, model: VQContinuousVAE):
        super().__init__()
        self.model = model

    @th.no_grad()
    def forward(self, traj_sequence: th.Tensor, terminals: th.Tensor) -> th.Tensor:
        return self.model.encode(traj_sequence, terminals)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = TransformerConfig(n_layers=4,
                               n_heads=4,
                               emb_dim=128,
                               traj_emb_dim=512,
                               latent_step=3,
                               n_tokens=512,
                               emb_dropout_rate=0.1,
                               resid_dropout_rate=0.1,
                               attn_dropout_rate=0.1,
                               pos_weight=1.0,
                               action_weight=5.0,
                               reward_weight=1.0,
                               value_weight=1.0,
                               first_action_weight=0.0,
                               sum_reward_weight=0.0,
                               last_value_weight=0.0,
                               bottleneck="attention",
                               masking="uniform",
                               residual=True,
                               ma_update=True,
                               observation_dim=17,
                               action_dim=6,
                               block_size=128,
                               transition_dim=17 + 6 + 1 + 1,
                               state_conditional=True)

    model = VQContinuousVAE(config)
    bs, seq_len = 1, 72
    state_sequence = th.rand(bs, seq_len, 17)
    action_sequence = th.rand(bs, seq_len, 6)
    reward_sequence = th.rand(bs, seq_len, 1)
    value_sequence = th.rand(bs, seq_len, 1)
    traj_sequence = th.concat([state_sequence, action_sequence, reward_sequence, value_sequence], dim=-1)
    terminals = th.zeros(bs, seq_len, 1)

    indices = model.encode(traj_sequence, terminals)
    traj_pred = model.decode_from_indices(indices, state_sequence[:, 0])
    print(traj_pred.shape)
