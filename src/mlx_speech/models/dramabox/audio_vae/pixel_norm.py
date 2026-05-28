"""Parameter-less PixelNorm: ``x / sqrt(mean(x**2, channel_axis) + eps)``.

The dimension is the channel axis. In PyTorch (channel-first) that's dim=1;
in MLX (channel-last) it's the last axis.

Reference: `.references/DramaBox/ltx2/ltx_core/model/common/normalization.py:14-40`
"""

from __future__ import annotations

import mlx.core as mx


def pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Per-pixel RMS normalization along the channel axis (last in MLX).

    Computes the operation in fp32 for stability and casts back.
    """
    orig_dtype = x.dtype
    x32 = x.astype(mx.float32)
    mean_sq = mx.mean(x32 * x32, axis=-1, keepdims=True)
    rms = mx.sqrt(mean_sq + eps)
    out = x32 / rms
    return out.astype(orig_dtype)


__all__ = ["pixel_norm"]
