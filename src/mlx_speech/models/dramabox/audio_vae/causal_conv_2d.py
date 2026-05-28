"""Causal Conv2d along the height (time) axis for audio mel spectrograms.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/causal_conv_2d.py`

For `causality_axis = "height"` (audio convention: height is the time axis):

    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)     # (L, R, T, B)

For kernel=3, dilation=1: ``pad_h = 2, pad_w = 2`` → ``pad = (1, 1, 2, 0)``.
The top is padded by ``2`` (causal: can only see past inputs), the height is
otherwise unchanged in size after the conv.

We store the inner Conv2d as ``self.conv`` so the saved key path
``some_module.conv.conv.weight`` lines up with the upstream checkpoint
naming (``CausalConv2d.conv`` is the actual `nn.Conv2d`).

MLX channel-last convention: input is ``[B, H, W, C]``. We translate from
PyTorch's ``[B, C, H, W]`` at the boundary.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CausalConv2d(nn.Module):
    """2D conv with causal padding along the height (time) axis.

    Pads input on top by ``(kernel_h - 1) * dilation`` (causal) and
    symmetrically on width. Strided convs are handled by passing
    ``stride > 1`` and a tightened pad — used by the encoder's Downsample.

    Args:
        in_channels, out_channels: PyTorch convention; we adapt to MLX's
            channel-last layout internally.
        kernel_size: int or (k_h, k_w).
        stride: int or (s_h, s_w). Default 1.
        dilation: int or (d_h, d_w). Default 1.
        bias: include bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        *,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation

        pad_h = (kh - 1) * dh
        pad_w = (kw - 1) * dw
        # PyTorch F.pad order is (left, right, top, bottom). MLX `mx.pad`
        # takes a per-axis ((before, after), ...) list. We store both forms.
        self._pad_left, self._pad_right = pad_w // 2, pad_w - pad_w // 2
        self._pad_top, self._pad_bottom = pad_h, 0
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.dilation = (dh, dw)

        # Inner Conv2d — MLX expects channel-last input.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(0, 0),
            dilation=(dh, dw),
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal pad along height, symmetric pad along width, then conv.

        Input: ``[B, H, W, C]`` (MLX channel-last).
        Output: ``[B, H_out, W_out, out_channels]`` per the conv stride.
        """
        # Pad on (H, W): top pad=pad_h, bottom pad=0; left=pad_w//2, right=pad_w-pad_w//2
        x = mx.pad(
            x,
            [(0, 0), (self._pad_top, self._pad_bottom), (self._pad_left, self._pad_right), (0, 0)],
        )
        return self.conv(x)


__all__ = ["CausalConv2d"]
