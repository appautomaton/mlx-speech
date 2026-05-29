"""PixArt-style combined timestep+size embedding.

Two-stage:
1. Sinusoidal embedding `get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True)`
2. Two-layer MLP `Linear(256 → hidden) → SiLU → Linear(hidden → hidden)`

Then a final ``linear`` projects to ``coeff * hidden`` for AdaLN.

Reference: `.references/DramaBox/ltx2/ltx_core/model/transformer/timestep_embedding.py`
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int = 256,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: int = 10_000,
) -> mx.array:
    """Standard diffusion-time sinusoidal embedding.

    Args:
        timesteps: float tensor with arbitrary leading dims — ``[B]`` (per-batch)
            or ``[B, T]`` (per-token, when timesteps vary by token under
            voice-ref conditioning).
    Returns:
        ``timesteps.shape + (embedding_dim,)`` — ``[B, embedding_dim]`` or
        ``[B, T, embedding_dim]``.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)  # [half_dim]
    # Broadcast the freq band against any leading shape: [..., 1] * [half_dim].
    emb = timesteps[..., None].astype(mx.float32) * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[..., half_dim:], emb[..., :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros_like(emb[..., :1])], axis=-1)
    return emb


class _TimestepMLP(nn.Module):
    """``Linear(256 → hidden) → SiLU → Linear(hidden → hidden)``.

    Saved keys: `linear_1.{weight, bias}` and `linear_2.{weight, bias}`.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden, bias=True)
        self.linear_2 = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class PixArtAlphaCombinedTimestepEmbedder(nn.Module):
    """Saved keys: ``emb.timestep_embedder.linear_{1,2}.{weight,bias}``."""

    def __init__(self, hidden: int):
        super().__init__()
        self.timestep_embedder = _TimestepMLP(hidden)

    def __call__(self, timesteps: mx.array, dtype: mx.Dtype) -> mx.array:
        proj = sinusoidal_timestep_embedding(timesteps)
        return self.timestep_embedder(proj.astype(dtype))


class AdaLayerNormSingle(nn.Module):
    """PixArt-Alpha ``AdaLayerNormSingle`` — produces ``coeff * hidden`` AdaLN
    factors from a timestep. Accepts a per-batch ``[B]`` timestep or a per-token
    ``[B, T]`` timestep (the latter for voice-ref conditioning, where reference
    tokens carry timestep 0); the MLP/linear act on the last axis, so the leading
    shape passes through.

    Saved keys:
        emb.timestep_embedder.linear_1.{weight, bias}
        emb.timestep_embedder.linear_2.{weight, bias}
        linear.{weight, bias}
    """

    def __init__(self, hidden: int, *, coeff: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepEmbedder(hidden)
        self.linear = nn.Linear(hidden, coeff * hidden, bias=True)

    def __call__(self, timesteps: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        """Returns ``(ada_factors, embedded_timestep)``.

        For ``[B]`` input: ``ada_factors`` is ``[B, coeff * hidden]`` and
        ``embedded_timestep`` is ``[B, hidden]``. For ``[B, T]`` input the leading
        ``T`` is preserved: ``[B, T, coeff * hidden]`` and ``[B, T, hidden]``.
        ``embedded_timestep`` (MLP output before the linear) feeds the model-level
        final-AdaLN.
        """
        t_emb = self.emb(timesteps, dtype)
        return self.linear(nn.silu(t_emb)), t_emb


__all__ = [
    "AdaLayerNormSingle",
    "PixArtAlphaCombinedTimestepEmbedder",
    "sinusoidal_timestep_embedding",
]
