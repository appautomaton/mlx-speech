"""Causal BigVGAN primitives for dots.tts waveform decoding."""

from __future__ import annotations

import math
from fractions import Fraction

import mlx.core as mx
import mlx.nn as nn
import numpy as np


HIGH_PRECISION_OUTPUT_TILE = 32
HIGH_PRECISION_TIME_TILE = 512


class Conv1d(nn.Module):
    """Channels-last Conv1d with explicit causal or symmetric padding."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = True,
        bias: bool = True,
        high_precision: bool = False,
    ):
        super().__init__()
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.kernel_size = int(kernel_size)
        self.high_precision = bool(high_precision)
        self.left_padding = self.dilation * (self.kernel_size - 1) if causal else 0
        self.padding = (
            0
            if causal
            else (self.kernel_size * self.dilation - self.dilation) // 2
        )
        scale = math.sqrt(2.0 / (input_channels * kernel_size + output_channels))
        self.weight = mx.random.normal(
            (output_channels, kernel_size, input_channels)
        ) * scale
        self.bias = mx.zeros((output_channels,)) if bias else None

    def __call__(self, value: mx.array) -> mx.array:
        if self.high_precision:
            return self._call_high_precision(value)
        if self.causal and self.left_padding:
            value = mx.pad(value, ((0, 0), (self.left_padding, 0), (0, 0)))
        output = mx.conv1d(
            value,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return output if self.bias is None else output + self.bias

    @property
    def left_context(self) -> int:
        if self.causal:
            return self.left_padding
        return self.padding

    @property
    def right_context(self) -> int:
        if self.causal:
            return 0
        receptive_field = self.dilation * (self.kernel_size - 1)
        return receptive_field - self.padding

    def _call_high_precision(self, value: mx.array) -> mx.array:
        """Run an encoder-only convolution with true FP32 reductions."""

        value = value.astype(mx.float32)
        if self.causal and self.left_padding:
            value = mx.pad(value, ((0, 0), (self.left_padding, 0), (0, 0)))
        elif self.padding:
            value = mx.pad(value, ((0, 0), (self.padding, self.padding), (0, 0)))
        weight = self.weight.astype(mx.float32)
        kernel = int(weight.shape[1])
        padded_time = int(value.shape[1])
        output_time = (
            padded_time - self.dilation * (kernel - 1) - 1
        ) // self.stride + 1
        time_tiles = []
        output_channels = int(weight.shape[0])
        for time_start in range(0, output_time, HIGH_PRECISION_TIME_TILE):
            time_end = min(time_start + HIGH_PRECISION_TIME_TILE, output_time)
            channel_tiles = []
            for channel_start in range(
                0, output_channels, HIGH_PRECISION_OUTPUT_TILE
            ):
                channel_end = min(
                    channel_start + HIGH_PRECISION_OUTPUT_TILE, output_channels
                )
                output = None
                for tap in range(kernel):
                    start = tap * self.dilation + time_start * self.stride
                    frames = value[
                        :,
                        start : start + self.stride * (time_end - time_start) : self.stride,
                        :,
                    ]
                    contribution = mx.sum(
                        frames[:, :, None, :]
                        * weight[
                            None,
                            None,
                            channel_start:channel_end,
                            tap,
                            :,
                        ],
                        axis=-1,
                    )
                    output = (
                        contribution if output is None else output + contribution
                    )
                if output is None:
                    raise ValueError("high-precision convolution has an empty kernel")
                if self.bias is not None:
                    output += self.bias[channel_start:channel_end].astype(mx.float32)
                channel_tiles.append(output)
            time_tile = mx.concatenate(channel_tiles, axis=-1)
            mx.eval(time_tile)
            time_tiles.append(time_tile)
        result = mx.concatenate(time_tiles, axis=1)
        mx.eval(result)
        return result


class CausalConvTranspose1d(nn.Module):
    """MLX-layout causal transposed convolution with exact stride trimming."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size != 2 * stride:
            raise ValueError("causal transposed convolution requires kernel_size=2*stride")
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        scale = math.sqrt(2.0 / (input_channels * kernel_size + output_channels))
        self.weight = mx.random.normal(
            (output_channels, kernel_size, input_channels)
        ) * scale
        self.bias = mx.zeros((output_channels,)) if bias else None

    def __call__(self, value: mx.array) -> mx.array:
        output = mx.conv_transpose1d(value, self.weight, stride=self.stride)
        if self.bias is not None:
            output += self.bias
        return output[:, : -self.stride]

    @property
    def left_context(self) -> int:
        overlap = self.kernel_size - self.stride
        return (overlap + self.stride - 1) // self.stride


def _default_filter(kernel_size: int, ratio: int) -> np.ndarray:
    position = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2
    cutoff = 0.5 / ratio
    sinc = 2 * cutoff * np.sinc(2 * cutoff * position)
    window = np.kaiser(kernel_size, beta=8.6)
    taps = sinc * window
    taps /= taps.sum()
    return taps.astype(np.float32)


class AliasFreeSnakeBeta(nn.Module):
    """Upsample → SnakeBeta → low-pass downsample alias-free activation."""

    def __init__(
        self,
        channels: int,
        *,
        ratio: int = 2,
        kernel_size: int = 12,
    ):
        super().__init__()
        if ratio <= 0 or kernel_size <= ratio:
            raise ValueError("alias-free resampler dimensions are invalid")
        self.ratio = int(ratio)
        taps = mx.array(_default_filter(kernel_size, ratio))
        self.up_filter = mx.broadcast_to(
            taps[None, :, None], (channels, kernel_size, 1)
        )
        self.down_filter = mx.broadcast_to(
            taps[None, :, None], (channels, kernel_size, 1)
        )
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, value: mx.array) -> mx.array:
        channels = int(value.shape[-1])
        upsampled = self.ratio * mx.conv_transpose1d(
            value.astype(mx.float32),
            self.up_filter,
            stride=self.ratio,
            groups=channels,
        )
        trim = int(self.up_filter.shape[1]) - self.ratio
        upsampled = upsampled[:, :-trim]
        alpha = mx.exp(self.alpha.astype(mx.float32))[None, None]
        beta = mx.exp(self.beta.astype(mx.float32))[None, None]
        activated = upsampled + mx.square(mx.sin(upsampled * alpha)) / (beta + 1e-9)
        left = int(self.down_filter.shape[1]) - 1
        padded = mx.concatenate(
            (
                mx.broadcast_to(
                    activated[:, :1],
                    (int(activated.shape[0]), left, channels),
                ),
                activated,
            ),
            axis=1,
        )
        return mx.conv1d(
            padded,
            self.down_filter,
            stride=self.ratio,
            groups=channels,
        ).astype(value.dtype)

    @property
    def left_context(self) -> int:
        upsample_context = int(self.up_filter.shape[1]) - 1
        downsample_context = int(self.down_filter.shape[1]) - 1
        return (
            upsample_context + downsample_context + self.ratio - 1
        ) // self.ratio


class AMPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ):
        super().__init__()
        self.convs1 = [
            Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                causal=True,
            )
            for dilation in dilations
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, causal=True)
            for _ in dilations
        ]
        self.activations = [
            AliasFreeSnakeBeta(channels) for _ in range(2 * len(dilations))
        ]

    def __call__(self, value: mx.array) -> mx.array:
        for index, (first, second) in enumerate(
            zip(self.convs1, self.convs2, strict=True)
        ):
            update = self.activations[2 * index](value)
            update = first(update)
            update = self.activations[2 * index + 1](update)
            value = value + second(update)
        return value

    @property
    def left_context(self) -> int:
        context = 0
        for index, (first, second) in enumerate(
            zip(self.convs1, self.convs2, strict=True)
        ):
            context += self.activations[2 * index].left_context
            context += first.left_context
            context += self.activations[2 * index + 1].left_context
            context += second.left_context
        return context


class BigVGANDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        initial_channels: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        lookahead: int = 2,
    ):
        super().__init__()
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("decoder upsample rates and kernels differ in length")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError("decoder residual kernels and dilations differ in length")
        self.lookahead = int(lookahead)
        self.conv_pre = Conv1d(
            latent_dim,
            initial_channels,
            2 * self.lookahead + 1,
            causal=False,
        )
        self.ups = []
        self.resblocks = []
        channels = int(initial_channels)
        for rate, kernel in zip(
            upsample_rates, upsample_kernel_sizes, strict=True
        ):
            output_channels = channels // 2
            self.ups.append(
                CausalConvTranspose1d(
                    channels,
                    output_channels,
                    kernel,
                    stride=rate,
                )
            )
            self.resblocks.append(
                [
                    AMPBlock(output_channels, residual_kernel, dilations)
                    for residual_kernel, dilations in zip(
                        resblock_kernel_sizes,
                        resblock_dilation_sizes,
                        strict=True,
                    )
                ]
            )
            channels = output_channels
        self.activation_post = AliasFreeSnakeBeta(channels)
        self.conv_post = Conv1d(channels, 1, 7, causal=True, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        value = self.conv_pre(value)
        for upsample, blocks in zip(self.ups, self.resblocks, strict=True):
            value = upsample(value)
            outputs = [block(value) for block in blocks]
            value = sum(outputs[1:], outputs[0]) / len(outputs)
        return mx.clip(self.conv_post(self.activation_post(value)), -1.0, 1.0)

    @property
    def stream_lookahead(self) -> int:
        return self.conv_pre.right_context

    @property
    def stream_left_context(self) -> int:
        context = Fraction(self.conv_pre.left_context)
        scale = Fraction(1)
        for upsample, blocks in zip(self.ups, self.resblocks, strict=True):
            context += scale * upsample.left_context
            scale /= upsample.stride
            context += scale * max(block.left_context for block in blocks)
        context += scale * self.activation_post.left_context
        context += scale * self.conv_post.left_context
        return math.ceil(context)

    def stream_window_size(self, maximum_chunk_size: int) -> int:
        if maximum_chunk_size <= 0:
            raise ValueError("maximum_chunk_size must be positive")
        return (
            int(maximum_chunk_size)
            + self.stream_lookahead
            + self.stream_left_context
        )


__all__ = [
    "AliasFreeSnakeBeta",
    "AMPBlock",
    "BigVGANDecoder",
    "CausalConvTranspose1d",
    "Conv1d",
    "HIGH_PRECISION_OUTPUT_TILE",
    "HIGH_PRECISION_TIME_TILE",
]
