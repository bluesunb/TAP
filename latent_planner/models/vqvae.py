import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, List, Optional, Tuple, Union


class VectorQuantization(th.autograd.Function):
    """
    Vector Quantization module

    Notes:
        - Input:
            1. inputs: (bs, seq_len, emb_dim) tensor of embeddings from the encoder
            2. codebook: (n_tokens, emb_dim) tensor of embeddings of the codebook

        - Output:
            1. indices: (bs, seq_len) tensor of indices of the nearest neighbors in the codebook
    """
    @staticmethod
    def forward(ctx: Any, inputs: th.Tensor, codebook: th.Tensor) -> th.Tensor:
        """
        Calculate the indices of the nearest neighbors in the codebook for each input vector.

        Returns:
            indices: (bs, seq_len) tensor of indices of the nearest neighbors in the codebook
        """
        with th.no_grad():
            emb_dim = codebook.size(1)
            inputs_size = inputs.size()     # (bs, seq_len, emb_dim)

            inputs_flat = inputs.view(-1, emb_dim)
            distances = th.cdist(inputs_flat, codebook, p=2)     # (bs * seq_len, n_tokens)
            indices = th.argmin(distances, dim=1)   # (batch_size * seq_len, )
            indices = indices.view(*inputs_size[:-1])   # (bs, seq_len)

            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(th.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: th.Tensor, codebook: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Select the nearest neighbors (codes) in the codebook for each input vector

        Returns:
            codes: (bs, seq_len, emb_dim) tensor of codes
            indices_flat: (bs * seq_len) flatten tensor of indices of the nearest neighbors in the codebook
        """
        indices = VectorQuantization.apply(inputs, codebook)
        indices_flat = indices.view(-1)  # (bs * seq_len, )
        ctx.save_for_backward(indices_flat, codebook)    # save indices_flat and codebook for backward pass
        ctx.mark_non_differentiable(indices_flat)

        # codes_flat: selected code from the codebook
        codes_flat = th.index_select(codebook, dim=0, index=indices_flat)     # (bs * seq_len, emb_dim)
        codes = codes_flat.view_as(inputs)      # (bs, seq_len, emb_dim)

        return codes, indices_flat

    @staticmethod
    def backward(ctx: Any, grad_output: th.Tensor, grad_indices: th.Tensor):
        """
        Calculate the gradient of the loss w.r.t. the inputs and the codebook.
        """
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()   # straight-through estimator
        if ctx.needs_input_grad[1]:
            indices_flat, codebook = ctx.saved_tensors
            emb_dim = codebook.size(1)

            grad_output_flat = grad_output.contiguous().view(-1, emb_dim)    # (bs * seq_len, emb_dim)
            grad_codebook = th.zeros_like(codebook)
            grad_codebook.index_add_(dim=0, index=indices_flat, source=grad_output_flat)  # straight-through estimator

        return grad_inputs, grad_codebook


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_tokens: int, emb_dim: int, decay: float = 0.99):
        """
        VQ Embedding that updates the codebook using Exponential Moving Average (EMA).

        Args:
            n_tokens (int): K in the paper. Number of tokens in the codebook.
            emb_dim (int): D in the paper. Dimension of the embedding.
        """

        super().__init__()
        embedding = th.zeros(n_tokens, emb_dim)
        embedding.uniform_(-1./n_tokens, 1./n_tokens)   # initialize embedding
        self.decay = decay

        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', th.ones(n_tokens))    # count of the number of updates to each embedding
        self.register_buffer('ema_weight', embedding.clone())   # exponential moving average of the embedding

    def forward(self, z_e_x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate discrete latent k for each embedding z_e_x.

        Args:
            z_e_x: (bs, seq_len, emb_dim): Embeddings from the encoder. = z_e(x)
        Returns:
            latents: (bs, seq_len, emb_dim): Discrete latent k.
        """
        z_e_x_ = z_e_x.contiguous()
        latents = VectorQuantization.apply(z_e_x_, self.embedding.weight)   # (bs, seq_len, emb_dim)
        return latents

    def straight_through(self, z_e_x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate code embeddings `z_q_x` and updated code embeddings `z_q_x_bar` for each embedding `z_e_x`.

        Args:
            z_e_x: (bs, seq_len, emb_dim): Embeddings from the encoder. = z_e(x)

        Returns:
            z_q_x: (bs, seq_len, emb_dim): Code embeddings.
            z_q_x_bar: (bs, seq_len, emb_dim): Updated code embeddings.
        """
        n_tokens, emb_dim = self.embedding.size()
        z_e_x = z_e_x.contiguous()     # (bs, seq_len, emb_dim)
        z_q_x, indices_flat = VectorQuantizationStraightThrough.apply(z_e_x, self.embedding)
        z_q_x = z_q_x.contiguous()     # (bs, seq_len, emb_dim)

        # Exponential moving average
        if self.training:
            encodings = F.one_hot(indices_flat, n_tokens).float()   # (bs * seq_len, n_tokens)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * th.sum(encodings, dim=0)  # (n_tokens, )

            # collect the sum of the embeddings for each latent token
            dw = encodings.transpose(1, 0) @ z_e_x.view(-1, emb_dim)   # (n_tokens, emb_dim)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            # normalize the embedding
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            self.embedding = self.embedding.detach()
            self.ema_weight = self.ema_weight.detach()
            self.ema_count = self.ema_count.detach()

        # new code for the updated embedding
        z_q_x_bar_flat = th.index_select(self.embedding, dim=0, index=indices_flat)   # (bs * seq_len, emb_dim)
        z_q_x_bar = z_q_x_bar_flat.view_as(z_e_x)     # (bs, seq_len, emb_dim)
        z_q_x_bar = z_q_x_bar.contiguous()

        return z_q_x, z_q_x_bar
    
    def __repr__(self):
        return f'VQEmbeddingEMA({self.embedding.size(0)}, {self.embedding.size(1)})'


class VQEmbedding(nn.Module):
    def __init__(self, n_token: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_token, emb_dim)
        self.embedding.weight.data.uniform_(-1./n_token, 1./n_token)

    def forward(self, z_e_x: th.Tensor):
        z_e_x = z_e_x.contiguous()
        latents = VectorQuantization.apply(z_e_x, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x: th.Tensor):
        """
        Return the code embeddings `z_q_x` and updated code embeddings `z_q_x_bar` for each embedding `z_e_x`.

        Args:
            z_e_x: (bs, seq_len, emb_dim): Embeddings from the encoder. = z_e(x)

        Returns:
            1) z_q_x: (bs, seq_len, emb_dim): Code embeddings.
            2) z_q_x_bar: (bs, seq_len, emb_dim): Updated code embeddings to calculate the VQ/Commitment loss with `z_e_x`.
        """
        z_e_x = z_e_x.contiguous()
        z_q_x, indices_flat = VectorQuantizationStraightThrough.apply(z_e_x, self.embedding.weight.detach())
        z_q_x = z_q_x.contiguous()

        z_q_x_bar_flat = th.index_select(self.embedding.weight, dim=0, index=indices_flat)
        z_q_x_bar = z_q_x_bar_flat.view_as(z_e_x)
        z_q_x_bar = z_q_x_bar.contiguous()

        return z_q_x, z_q_x_bar
    
    def __repr__(self):
        return f'VQEmbedding({self.embedding.num_embeddings}, {self.embedding.embedding_dim})'


if __name__ == "__main__":
    bs, seq_len, emb_dim = 2, 3, 4
    n_tokens = 5
    z_e_x = th.rand(bs, seq_len, emb_dim)

    vq = VQEmbedding(n_tokens, emb_dim)
    z_q_x, z_q_x_bar = vq.straight_through(z_e_x)
    print(z_q_x.shape)
    print(z_q_x_bar.shape)