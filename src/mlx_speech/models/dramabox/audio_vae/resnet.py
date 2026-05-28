"""VAE ResnetBlock with PixelNorm + SiLU + CausalConv2d.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/resnet.py:115-176`

Saved keys (per block):
    conv1.conv.{weight,bias}        (CausalConv2d → Conv2d)
    conv2.conv.{weight,bias}
    nin_shortcut.conv.{weight,bias}  (present iff in != out channels)

No `norm1.weight` / `norm2.weight` since PixelNorm is parameter-less.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .causal_conv_2d import CausalConv2d
from .pixel_norm import pixel_norm


class ResnetBlock(nn.Module):
    """VAE ResnetBlock: ``norm → SiLU → conv1 → norm → SiLU → conv2 + skip``.

    Args:
        in_channels: input channel count.
        out_channels: output channel count. If different from ``in_channels``,
            a ``nin_shortcut`` (1×1 causal conv) projects the skip.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=True)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        h = pixel_norm(x)
        h = nn.silu(h)
        h = self.conv1(h)

        h = pixel_norm(h)
        h = nn.silu(h)
        h = self.conv2(h)

        skip = x if self.nin_shortcut is None else self.nin_shortcut(x)
        return skip + h


__all__ = ["ResnetBlock"]
