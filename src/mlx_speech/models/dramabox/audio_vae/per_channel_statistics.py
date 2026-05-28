"""Per-channel statistics buffers (mean-of-means, std-of-means).

Used to (un-)normalize patchified latents around encoder output / decoder
input. Shape ``[channels * mel_bins] = [128]`` for DramaBox (8 × 16).

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/ops.py:58-73`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PerChannelStatistics(nn.Module):
    """Latent normalization buffers.

    Saved keys (note: keys use a hyphen, not an underscore):
        mean-of-means: [128]
        std-of-means:  [128]
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        # Use attribute names that MLX can serialize then patch via load_weights
        # remap. The saved keys use hyphens which aren't valid Python attrs,
        # so we use Python-legal names and remap at load time.
        self.mean_of_means = mx.zeros((dim,), dtype=mx.float32)
        self.std_of_means = mx.ones((dim,), dtype=mx.float32)

    def normalize(self, x: mx.array) -> mx.array:
        """``(x - mean) / std`` along the patched last dim."""
        mean = self.mean_of_means.astype(x.dtype)
        std = self.std_of_means.astype(x.dtype)
        return (x - mean) / std

    def un_normalize(self, x: mx.array) -> mx.array:
        """``x * std + mean`` (inverse of `normalize`)."""
        mean = self.mean_of_means.astype(x.dtype)
        std = self.std_of_means.astype(x.dtype)
        return x * std + mean


__all__ = ["PerChannelStatistics"]
