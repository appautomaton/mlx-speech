"""SEMamba speech-enhancement model (pure-MLX port of nvidia/RE-USE).

Mirrors `.references/RE-USE/models/generator_SEMamba_time_d4.py` and
`.references/RE-USE/models/codec_module_time_d4.py`. The bidirectional Mamba
blocks live in `mamba/block.py`; this file assembles the dense encoder, the 30
time-frequency Mamba blocks, and the magnitude / phase decoders.

MLX convolutions use channel-last tensors ``[B, H, W, C]`` while the torch
reference is channel-first ``[B, C, H, W]``. To keep the port faithful and the
weight remap mechanical, every conv module here stores its weight in MLX layout
``[out, kh, kw, in]`` (Conv2d) and transposes the feature map in/out around each
conv call. Public forward I/O stays channel-first, matching the reference.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .mamba.block import TFMambaBlock


def _to_channels_last(x: mx.array) -> mx.array:
    """``[B, C, H, W]`` -> ``[B, H, W, C]`` for an MLX conv call."""
    return mx.transpose(x, (0, 2, 3, 1))


def _to_channels_first(x: mx.array) -> mx.array:
    """``[B, H, W, C]`` -> ``[B, C, H, W]`` after an MLX conv call."""
    return mx.transpose(x, (0, 3, 1, 2))


def _get_padding_2d(kernel_size, dilation=(1, 1)) -> tuple[int, int]:
    """Same-padding helper. Mirrors `codec_module_time_d4.py:get_padding_2d`."""
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class _ConvNormAct(nn.Module):
    """Conv2d -> InstanceNorm2d(affine) -> PReLU on channel-first features.

    Holds the three submodules under the names ``.0`` / ``.1`` / ``.2`` is not
    valid Python, so we expose them as a small indexable container matching the
    torch ``nn.Sequential`` key layout (``<name>.0.weight`` etc.).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        stride=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.InstanceNorm(out_channels, affine=True)
        self.act = nn.PReLU(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x channel-first; conv + norm run channel-last, then back.
        h = self.conv(_to_channels_last(x))
        h = self.norm(h)
        h = _to_channels_first(h)
        # PReLU weight is per-channel; apply on channel-first to broadcast over C.
        w = self.act.weight.reshape(1, -1, 1, 1)
        return mx.where(h >= 0, h, w * h)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel transpose conv along the frequency axis.

    Mirrors `codec_module_time_d4.py:SPConvTranspose2d`: pad F by (1, 1), Conv2d
    to ``out_channels * r`` channels, then pixel-shuffle the r factor into F.
    Operates on channel-first ``[B, C, H, W]``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.r = r
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size, stride=(1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        # Pad W (frequency) by 1 on each side, like nn.ConstantPad2d((1,1,0,0)).
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 1)])
        out = _to_channels_first(self.conv(_to_channels_last(x)))
        b, nch, h, w = out.shape
        out = out.reshape(b, self.r, nch // self.r, h, w)
        out = mx.transpose(out, (0, 2, 3, 4, 1))
        out = out.reshape(b, nch // self.r, h, -1)
        return out


class DenseBlock(nn.Module):
    """Dilated dense block (depth 4). Mirrors `codec_module_time_d4.py:DenseBlock`."""

    def __init__(self, hid_feature: int, kernel_size=(3, 3), depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.dense_block = [
            _ConvNormAct(
                hid_feature * (i + 1),
                hid_feature,
                kernel_size,
                dilation=(2**i, 1),
                padding=_get_padding_2d(kernel_size, (2**i, 1)),
            )
            for i in range(depth)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        skip = x
        out = x
        for i in range(self.depth):
            out = self.dense_block[i](skip)
            skip = mx.concatenate([out, skip], axis=1)
        return out


class DenseEncoder(nn.Module):
    """Dense encoder. Mirrors `codec_module_time_d4.py:DenseEncoder`."""

    def __init__(self, input_channel: int, hid_feature: int) -> None:
        super().__init__()
        self.dense_conv_1 = _ConvNormAct(input_channel, hid_feature, (1, 1))
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.dense_conv_2 = _ConvNormAct(
            hid_feature, hid_feature, (1, 3), stride=(4, 2)
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MagDecoder(nn.Module):
    """Magnitude decoder. Mirrors `codec_module_time_d4.py:MagDecoder`."""

    def __init__(self, hid_feature: int, output_channel: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.up_conv1 = _SPUp(hid_feature, hid_feature, r=2)
        self.up_conv2 = _SPUp(hid_feature, hid_feature, r=4)
        self.final_conv = nn.Conv2d(hid_feature, output_channel, (1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense_block(x)
        x = self.up_conv1(x)
        # up_conv2 runs on the H<->W transposed map, then transposes back.
        x = self.up_conv2(mx.transpose(x, (0, 1, 3, 2)))
        x = mx.transpose(x, (0, 1, 3, 2))
        x = _to_channels_first(self.final_conv(_to_channels_last(x)))
        return x


class PhaseDecoder(nn.Module):
    """Phase decoder. Mirrors `codec_module_time_d4.py:PhaseDecoder`."""

    def __init__(self, hid_feature: int, output_channel: int) -> None:
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.up_conv1 = _SPUp(hid_feature, hid_feature, r=2)
        self.up_conv2 = _SPUp(hid_feature, hid_feature, r=4)
        self.phase_conv_r = nn.Conv2d(hid_feature, output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid_feature, output_channel, (1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(mx.transpose(x, (0, 1, 3, 2)))
        x = mx.transpose(x, (0, 1, 3, 2))
        x_r = _to_channels_first(self.phase_conv_r(_to_channels_last(x)))
        x_i = _to_channels_first(self.phase_conv_i(_to_channels_last(x)))
        return mx.arctan2(x_i, x_r)


class _SPUp(nn.Module):
    """SPConvTranspose2d -> InstanceNorm2d(affine) -> PReLU (an up_conv stage).

    Matches the torch ``nn.Sequential`` key layout ``up_convN.0.conv.*`` /
    ``up_convN.1.*`` (norm) / ``up_convN.2.weight`` (PReLU).
    """

    def __init__(self, in_channels: int, out_channels: int, r: int) -> None:
        super().__init__()
        self.conv = SPConvTranspose2d(in_channels, out_channels, (1, 3), r=r)
        self.norm = nn.InstanceNorm(out_channels, affine=True)
        self.act = nn.PReLU(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv(x)  # channel-first
        h = _to_channels_first(self.norm(_to_channels_last(h)))
        w = self.act.weight.reshape(1, -1, 1, 1)
        return mx.where(h >= 0, h, w * h)


class SEMamba(nn.Module):
    """SEMamba generator. Mirrors `generator_SEMamba_time_d4.py:SEMamba`.

    Forward takes channel-first ``noisy_mag`` / ``noisy_pha`` ``[B, F, T]`` and
    returns ``(denoised_mag, denoised_pha, denoised_com)`` with the same F, T.
    """

    def __init__(
        self,
        *,
        num_tfmamba: int = 30,
        hid_feature: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        input_channel: int = 2,
        output_channel: int = 1,
    ) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(input_channel, hid_feature)
        self.TSMamba = [
            TFMambaBlock(hid_feature, d_state, d_conv, expand)
            for _ in range(num_tfmamba)
        ]
        self.mask_decoder = MagDecoder(hid_feature, output_channel)
        self.phase_decoder = PhaseDecoder(hid_feature, output_channel)

    def __call__(
        self, noisy_mag: mx.array, noisy_pha: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        # [B, F, T] -> [B, 1, T, F] each, then concat on channel -> [B, 2, T, F].
        noisy_mag = mx.transpose(noisy_mag, (0, 2, 1))[:, None, :, :]
        noisy_pha = mx.transpose(noisy_pha, (0, 2, 1))[:, None, :, :]
        x = mx.concatenate([noisy_mag, noisy_pha], axis=1)  # [B, 2, T, F]

        # Anti-error pad: +2 on F then +2 on T (matches the reference order).
        b, c, t, f = x.shape
        x = mx.concatenate([x, mx.zeros((b, c, t, 2), dtype=x.dtype)], axis=-1)
        x = mx.concatenate([x, mx.zeros((b, c, 2, f + 2), dtype=x.dtype)], axis=-2)

        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)

        # Decode -> [B, C, T, F] -> [B, F, T, C] -> squeeze C.
        denoised_mag = mx.transpose(self.mask_decoder(x), (0, 3, 2, 1))[..., 0]
        denoised_pha = mx.transpose(self.phase_decoder(x), (0, 3, 2, 1))[..., 0]

        denoised_mag = denoised_mag[:, :f, :t]
        denoised_pha = denoised_pha[:, :f, :t]

        denoised_com = mx.stack(
            [denoised_mag * mx.cos(denoised_pha), denoised_mag * mx.sin(denoised_pha)],
            axis=-1,
        )
        return denoised_mag, denoised_pha, denoised_com


__all__ = [
    "SEMamba",
    "DenseEncoder",
    "DenseBlock",
    "MagDecoder",
    "PhaseDecoder",
    "SPConvTranspose2d",
]
