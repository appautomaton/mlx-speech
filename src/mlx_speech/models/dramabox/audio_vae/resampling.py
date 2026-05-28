"""Downsample (encoder) and Upsample (decoder) modules.

Both use a kernel-3 stride-1-or-2 `CausalConv2d` with an explicit asymmetric
pad. The upsample first does nearest-neighbor 2× interpolation, then runs
the conv, then drops the first row of the height axis to undo the encoder's
top padding.

Reference:
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/downsample.py`
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/upsample.py`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Downsample(nn.Module):
    """Strided ``Conv2d`` with causality-aware asymmetric padding.

    For ``causality_axis = "height"`` the upstream pad is
    ``(left=0, right=1, top=2, bottom=0)`` and the stride is 2 in both axes.
    The internal conv has kernel 3, stride 2, padding 0.

    Saved key: ``downsample.conv.{weight, bias}``.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # MLX channel-last: pad spec is per-axis ((before, after), ...)
        # For (B, H, W, C): pad H = (top=2, bottom=0), W = (left=0, right=1)
        x = mx.pad(x, [(0, 0), (2, 0), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2× upsample + ``CausalConv2d``, then drop the first
    row along the time (height) axis.

    The upstream causal Upsample stores ``self.conv = CausalConv2d`` so the
    saved key is ``upsample.conv.conv.{weight,bias}``.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Save under `conv` as a CausalConv2d to mirror upstream key path.
        from .causal_conv_2d import CausalConv2d
        self.conv = CausalConv2d(in_channels, in_channels, kernel_size=3, stride=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # Nearest-neighbor 2× upsample along H and W (height, width).
        # MLX has no direct interpolate; use a repeat-then-reshape trick.
        # Input shape (B, H, W, C). We want (B, 2H, 2W, C).
        B, H, W, C = x.shape
        # Repeat each row twice along H, then each col twice along W.
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        # Now shape is (B, 2H, 2W, C)
        x = self.conv(x)
        # Drop the first row along H (causality_axis = height in upstream code).
        x = x[:, 1:, :, :]
        return x


__all__ = ["Downsample", "Upsample"]
