"""BigVGAN-v2 generator: ``Vocoder`` class.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/vocoder.py:271-416`

Architecture (the same code structure for both main + BWE):

    conv_pre   : Conv1d(in_channels=128, out=ch0, kernel=7)
    for i in range(num_upsamples):
        x = ups[i](x)                          # ConvTranspose1d
        for k in range(num_kernels):
            blocks[i*K + k](x)                 # AMPBlock1
        x = mean(block_outputs)
    act_post   : Activation1d(SnakeBeta(final_ch))
    conv_post  : Conv1d(in=final_ch, out=2, kernel=7, bias=use_bias_at_final)
    optional final tanh / clip if `apply_final_activation`

Saved keys mirror this layout exactly; we use Python lists for `ups` and
`resblocks` so MLX serializes them as ``ups.0``, ``ups.1``, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .anti_aliased import Activation1d
from .snake import SnakeBeta


@dataclass(frozen=True)
class VocoderArgs:
    """Configuration of one Vocoder generator (main or BWE)."""

    upsample_initial_channel: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, int, int], ...] = (
        (1, 3, 5), (1, 3, 5), (1, 3, 5),
    )
    in_channels: int = 128
    out_channels: int = 2
    activation: str = "snakebeta"
    use_tanh_at_final: bool = False
    apply_final_activation: bool = True
    use_bias_at_final: bool = False

    @property
    def num_upsamples(self) -> int:
        return len(self.upsample_rates)

    @property
    def num_kernels(self) -> int:
        return len(self.resblock_kernel_sizes)

    @property
    def final_channels(self) -> int:
        ch = self.upsample_initial_channel
        for _ in self.upsample_rates:
            ch //= 2
        return ch


# --------------------------------------------------------------------------- #
# AMPBlock1
# --------------------------------------------------------------------------- #

class AMPBlock1(nn.Module):
    """3 conv pairs with anti-aliased SnakeBeta activations.

    Forward: for each `(c1, c2, a1, a2)` triple:
        x = x + c2(a2(c1(a1(x))))

    Saved keys per block:
        convs1.{0..2}.{weight, bias}    Conv1d, dilated
        convs2.{0..2}.{weight, bias}    Conv1d, dilation=1
        acts1.{0..2}.{act.alpha, act.beta, upsample.filter, downsample.lowpass.filter}
        acts2.{0..2}.{...}
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        # convs1: dilated 1D convs (padding chosen so output T == input T)
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d,
                      padding=_get_padding(kernel_size, d), bias=True)
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1,
                      padding=_get_padding(kernel_size, 1), bias=True)
            for _ in dilation
        ]
        self.acts1 = [Activation1d(channels) for _ in dilation]
        self.acts2 = [Activation1d(channels) for _ in dilation]

    def __call__(self, x: mx.array) -> mx.array:
        """Input/output shape: ``(B, C, T)``."""
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = _conv1d_channels_first(c1, xt)
            xt = a2(xt)
            xt = _conv1d_channels_first(c2, xt)
            x = x + xt
        return x


# --------------------------------------------------------------------------- #
# Vocoder generator
# --------------------------------------------------------------------------- #

class Vocoder(nn.Module):
    """BigVGAN-v2 generator (mel → wav).

    Forward shape: ``(B, 2, T, mel_bins)`` (stereo) or ``(B, T, mel_bins)`` (mono)
    → output ``(B, out_channels, T_wav)``.
    """

    def __init__(self, args: VocoderArgs):
        super().__init__()
        self.args = args

        ch = args.upsample_initial_channel
        self.conv_pre = nn.Conv1d(args.in_channels, ch, 7, stride=1, padding=3, bias=True)

        # Upsample stages — each is a ConvTranspose1d
        self.ups: list[nn.ConvTranspose1d] = []
        for stride, ks in zip(args.upsample_rates, args.upsample_kernel_sizes):
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels=ch,
                    out_channels=ch // 2,
                    kernel_size=ks,
                    stride=stride,
                    padding=(ks - stride) // 2,
                    bias=True,
                )
            )
            ch //= 2

        # Resblocks — num_upsamples × num_kernels (one AMPBlock1 per resblock_kernel_size
        # at each upsample stage)
        self.resblocks: list[AMPBlock1] = []
        ch = args.upsample_initial_channel
        for _ in range(args.num_upsamples):
            ch //= 2
            for k, d in zip(args.resblock_kernel_sizes, args.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d))

        # Final activation + conv
        self.act_post = Activation1d(args.final_channels)
        self.conv_post = nn.Conv1d(
            args.final_channels, args.out_channels, 7, stride=1, padding=3,
            bias=args.use_bias_at_final,
        )

    def __call__(self, mel: mx.array) -> mx.array:
        """Forward.

        Input: 4D ``(B, 2, T, mel_bins)`` for stereo, else 3D ``(B, T, mel_bins)``.
        Output: ``(B, out_channels, T_wav)``.
        """
        # Transpose so time is the last axis: (B, C, T, F) → (B, C, F, T)
        x = mel.transpose(0, 1, 3, 2) if mel.ndim == 4 else mel.transpose(0, 2, 1)
        # Combine stereo + mel into channels: (B, 2, F, T) → (B, 2*F, T)
        if mel.ndim == 4:
            B, S, F, T = x.shape
            x = x.reshape(B, S * F, T)

        x = _conv1d_channels_first(self.conv_pre, x)

        for i in range(self.args.num_upsamples):
            x = _convtranspose1d_channels_first(self.ups[i], x)
            start = i * self.args.num_kernels
            end = start + self.args.num_kernels
            outs = []
            for idx in range(start, end):
                outs.append(self.resblocks[idx](x))
            stacked = mx.stack(outs, axis=0)
            x = mx.mean(stacked, axis=0)

        x = self.act_post(x)
        x = _conv1d_channels_first(self.conv_post, x)

        if self.args.apply_final_activation:
            if self.args.use_tanh_at_final:
                x = mx.tanh(x)
            else:
                x = mx.clip(x, -1.0, 1.0)
        return x


# --------------------------------------------------------------------------- #
# Helpers — channel-first ⇆ MLX Conv1d (channel-last)
# --------------------------------------------------------------------------- #

def _conv1d_channels_first(conv: nn.Conv1d, x: mx.array) -> mx.array:
    """Apply a `Conv1d` to a channel-first tensor (B, C, T) and return (B, C', T')."""
    return conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def _convtranspose1d_channels_first(conv: nn.ConvTranspose1d, x: mx.array) -> mx.array:
    """Apply a `ConvTranspose1d` to a channel-first tensor (B, C, T)."""
    return conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


__all__ = ["Vocoder", "VocoderArgs", "AMPBlock1"]
