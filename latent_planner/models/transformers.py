import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any

from latent_planner.config import TransformerConfig


class SelfAttention(nn.Module):
    """
    Simple Self-Attention Module

    Attributes:
        key (nn.Linear):                Key projection
        query (nn.Linear):              Query projection.
        value (nn.Linear):              Value projection.
        attn_drop (nn.Dropout):         Dropout layer for attention weights.
        resid_drop (nn.Dropout):        Dropout layer for residual connection.
        out_proj (nn.Linear):           Output projection.
        n_head (int):                   Number of attention heads.
        _attn_map (Optional[th.Tensor]):    Attention map.

    Notes:
        - input of `forward` method is expected to be of shape (batch_size, seq_len, emb_dim)
    """
    def __init__(self, config: TransformerConfig, causal: bool = False):
        """
        Args:
            causal (bool):                  Whether to use causal mask.
        """
        super().__init__()
        assert config.emb_dim % config.n_heads == 0, \
            f"emb_dim ({config.emb_dim}) must be divisible by n_head ({config.n_heads})"

        self.config = config
        # Key, query, value projections for all heads
        self.key = nn.Linear(config.emb_dim, config.emb_dim)
        self.query = nn.Linear(config.emb_dim, config.emb_dim)
        self.value = nn.Linear(config.emb_dim, config.emb_dim)

        # Regularization
        self.attn_drop = nn.Dropout(config.attn_dropout_rate)
        self.resid_drop = nn.Dropout(config.resid_dropout_rate)

        # Output projection
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)

        # Causal mask
        if causal:
            mask = th.tril(th.ones(config.block_size, config.block_size))
            if hasattr(config, "action_dim"):
                joined_dim = config.observation_dim + config.action_dim + 2     # 2 for reward and value
                mask[:, joined_dim - 1::joined_dim] = 0     # mask previous value estimates
        else:
            mask = th.zeros(config.block_size, config.block_size)

        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))

        self.n_head = config.n_heads
        self._attn_map = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (bs, seq_len, emb_dim): Intermediate input sequence of embeddings.
        Returns:
            (bs, seq_len, emb_dim): Attention applied output sequence of embeddings.
        """
        seq_len = x.size(1)
        # k,q,v : (bs, n_head, seq_len, head_dim = emb_dim // n_head)
        k = rearrange(self.key(x), "b t (h d) -> b h t d", h=self.n_head)
        q = rearrange(self.query(x), "b t (h d) -> b h t d", h=self.n_head)
        v = rearrange(self.value(x), "b t (h d) -> b h t d", h=self.n_head)

        # attn_weights : (bs, n_head, seq_len, seq_len)
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._attn_map = attn_weights.clone()
        attn_weights = self.attn_drop(attn_weights)

        # attn_values : (bs, n_head, seq_len, head_dim)
        attn_values = attn_weights @ v
        attn_values = rearrange(attn_values, "b h t d -> b t (h d)")
        attn_values = self.resid_drop(self.out_proj(attn_values))
        return attn_values


class AttentionBlock(nn.Module):
    """
    Attention block that consists of self-attention and MLP with residual connection.

    Attributes:
        ln1 (nn.LayerNorm):     Layer normalization for self-attention.
        ln2 (nn.LayerNorm):     Layer normalization for MLP.
        attn (SelfAttention):   Self-attention module.
        mlp (nn.Sequential):    MLP module.
    """
    def __init__(self, config: TransformerConfig):
        """
        Notes:
            - inputs: (bs, seq_len, emb_dim)
            - outputs: (bs, seq_len, emb_dim)
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.attn = SelfAttention(config, causal=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            nn.GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
            nn.Dropout(config.resid_dropout_rate),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (bs, seq_len, emb_dim): Intermediate input sequence of embeddings.
        Returns:
            (bs, seq_len, emb_dim): Attention applied output sequence of embeddings.
        """
        x = x + self.attn(self.ln1(x))  # (bs, seq_len, emb_dim)
        x = x + self.mlp(self.ln2(x))   # (bs, seq_len, emb_dim)
        return x


class AsymAttentionBlock(nn.Module):
    """
    Asymmetric attention block that consists of self-attention and MLP with residual connection.
    This block changes the sequence length to `out_tokens`.

    - inputs: (bs, seq_len, emb_dim)
    - outputs: (bs, out_tokens, emb_dim)
    """
    def __init__(self, config: TransformerConfig, out_tokens: int):
        super().__init__()
        self.key = nn.Linear(config.emb_dim, config.emb_dim)
        self.query = nn.Parameter(th.rand(1, out_tokens, config.emb_dim))
        self.value = nn.Linear(config.emb_dim, config.emb_dim)

        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.attention = nn.MultiheadAttention(config.emb_dim, config.n_heads, batch_first=True)
        self.mlp = nn.Linear(config.emb_dim, config.emb_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (bs, seq_len, emb_dim)
        Returns:
            (bs, out_tokens, emb_dim)
        Notes:
            `out_tokens` is the number of tokens in the output sequence.
            Note that it is different from standard attention block
            where `seq_len` is the number of tokens in the output sequence.
        """
        x = self.ln1(x)

        k = self.key(x)     # (bs, seq_len, emb_dim)
        v = self.value(x)    # (bs, seq_len, emb_dim)
        q = self.query.repeat([x.size(0), 1, 1])    # (bs, out_tokens, emb_dim)

        # attn_values: (bs, out_tokens, emb_dim)
        # attn_weights: (bs, out_tokens, seq_len)
        attn_values, attn_weights = self.attention(q, k, v)

        out = self.mlp(self.ln2(attn_values))   # (bs, out_tokens, emb_dim)
        return out
