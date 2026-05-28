"""Snake / SnakeBeta activation with per-channel learnable params.

SnakeBeta formula (used by DramaBox's BigVGAN-v2 generators):

    out = x + (1 / (beta + eps)) * sin(x * alpha) ** 2

`alpha` and `beta` are per-channel parameters. With `alpha_logscale=True`
(the default in upstream LTX code) they are stored as log-values: actual
multipliers used at inference are `exp(alpha)`, `exp(beta)`.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/vocoder.py:186-208`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SnakeBeta(nn.Module):
    """SnakeBeta with per-channel ``alpha`` (frequency) and ``beta`` (amplitude)."""

    def __init__(self, channels: int, *, alpha_logscale: bool = True):
        super().__init__()
        self.alpha = mx.zeros((channels,), dtype=mx.float32)
        self.beta = mx.zeros((channels,), dtype=mx.float32)
        self.alpha_logscale = alpha_logscale
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """x shape: ``(B, C, T)`` — channel-axis is dim 1 in vocoder convention."""
        a = self.alpha.astype(x.dtype)
        b = self.beta.astype(x.dtype)
        if self.alpha_logscale:
            a = mx.exp(a)
            b = mx.exp(b)
        # (C,) broadcast to (1, C, 1) for (B, C, T) input
        a = a[None, :, None]
        b = b[None, :, None]
        return x + (1.0 / (b + self.eps)) * mx.sin(x * a) ** 2


__all__ = ["SnakeBeta"]
