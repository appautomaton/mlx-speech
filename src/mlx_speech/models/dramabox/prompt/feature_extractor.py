"""DramaBox `FeatureExtractorV2` — audio-only path.

Stacks all 49 Gemma hidden states, applies per-token RMS normalization
across the embedding dimension, reshapes to flat features, rescales by
``sqrt(out_features / embedding_dim)``, and projects through
``audio_aggregate_embed`` (a `Linear(188160 → 2048)` held at bf16).

Reference: `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/feature_extractor.py:112-141`

The upstream class also supports a video projection. DramaBox is audio-only:
the upstream checkpoint ships no `video_aggregate_embed`, so we omit that
branch entirely.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class FeatureExtractorV2(nn.Module):
    """Per-token RMS norm → rescale → audio_aggregate_embed.

    Args:
        embedding_dim: per-layer hidden size of the Gemma backbone (3840).
        out_features: output width of the projection (2048 for DramaBox).

    Saved keys (at module init we let MLX manage `audio_aggregate_embed.{weight, bias}`):
        audio_aggregate_embed.weight: [2048, 188160]
        audio_aggregate_embed.bias:   [2048]
    """

    def __init__(self, embedding_dim: int, out_features: int, num_layers: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.num_layers = num_layers
        in_features = embedding_dim * num_layers
        self.audio_aggregate_embed = nn.Linear(in_features, out_features, bias=True)
        # Precompute the rescale constant
        self._rescale = math.sqrt(out_features / embedding_dim)

    def __call__(
        self,
        hidden_states: list[mx.array] | tuple[mx.array, ...],
        attention_mask: mx.array,
    ) -> mx.array:
        """Forward.

        Args:
            hidden_states: list/tuple of ``num_layers`` tensors each of shape
                ``[B, T, embedding_dim]``.
            attention_mask: ``[B, T]`` binary mask (1 = valid, 0 = pad).

        Returns:
            ``[B, T, out_features]`` audio feature tensor in the hidden states'
            compute dtype, with padded positions zeroed.
        """
        # Stack to [B, T, D, L]
        encoded = mx.stack(list(hidden_states), axis=-1)
        orig_dtype = encoded.dtype

        # Per-token RMS norm across D only: variance over axis 2
        # (the variance is shared across layers L, just like the upstream code).
        x32 = encoded.astype(mx.float32)
        variance = mx.mean(x32 * x32, axis=2, keepdims=True)  # [B, T, 1, L]
        normed = x32 * mx.rsqrt(variance + 1e-6)
        # Flatten to [B, T, D*L]
        B, T, D, L = encoded.shape
        normed = normed.reshape(B, T, D * L)

        # Zero out padded positions
        mask3d = attention_mask.astype(mx.bool_).reshape(B, T, 1)
        normed = mx.where(mask3d, normed, mx.zeros_like(normed))

        # Cast back to compute dtype, rescale, project
        normed = normed.astype(orig_dtype)
        rescaled = normed * mx.array(self._rescale, dtype=orig_dtype)
        out = self.audio_aggregate_embed(rescaled)
        return out


__all__ = ["FeatureExtractorV2"]
