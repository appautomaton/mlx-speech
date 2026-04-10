"""MLX-native WAV-VAE for LongCat AudioDiT."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import LongCatVaeConfig


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _ChannelFirstConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = (in_channels * kernel_size) ** -0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,), dtype=self.weight.dtype)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv1d(x, self.weight, self.stride, self.padding, self.dilation, 1)
        if "bias" in self:
            y = y + self.bias
        return mx.transpose(y, (0, 2, 1))


class _ChannelFirstConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = (in_channels * kernel_size) ** -0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,), dtype=self.weight.dtype)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv_transpose1d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return mx.transpose(y, (0, 2, 1))


def _snake_beta(x: mx.array, alpha: mx.array, beta: mx.array) -> mx.array:
    return x + ((1.0 / (beta + 1e-9)) * mx.square(mx.sin(x * alpha)))


class LongCatSnakeBeta(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((in_features,), dtype=mx.float32)
        self.beta = mx.zeros((in_features,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)[None, :, None]
        beta = mx.exp(self.beta)[None, :, None]
        return _snake_beta(x, alpha, beta)


def _get_vae_activation(use_snake: bool, channels: int) -> nn.Module:
    return LongCatSnakeBeta(channels) if use_snake else nn.ELU()


def _pixel_unshuffle_1d(x: mx.array, factor: int) -> mx.array:
    batch, channels, width = x.shape
    reshaped = x.reshape(batch, channels, width // factor, factor)
    shuffled = mx.transpose(reshaped, (0, 1, 3, 2))
    return shuffled.reshape(batch, channels * factor, width // factor)


def _pixel_shuffle_1d(x: mx.array, factor: int) -> mx.array:
    batch, channels, width = x.shape
    channels = channels // factor
    reshaped = x.reshape(batch, channels, factor, width)
    shuffled = mx.transpose(reshaped, (0, 1, 3, 2))
    return shuffled.reshape(batch, channels, width * factor)


class _DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.group_size = (in_channels * factor) // out_channels
        self.out_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        x = _pixel_unshuffle_1d(x, self.factor)
        batch, _, width = x.shape
        x = x.reshape(batch, self.out_channels, self.group_size, width)
        return mx.mean(x, axis=2)


class _UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.repeats = (out_channels * factor) // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, self.repeats, axis=1)
        return _pixel_shuffle_1d(x, self.factor)


class _VaeResidualUnit(nn.Module):
    def __init__(self, channels: int, *, dilation: int, use_snake: bool) -> None:
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = [
            _get_vae_activation(use_snake, channels),
            _ChannelFirstConv1d(
                channels, channels, 7, padding=padding, dilation=dilation, bias=True
            ),
            _get_vae_activation(use_snake, channels),
            _ChannelFirstConv1d(channels, channels, 1, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return residual + x


class _VaeEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        use_snake: bool,
        downsample_shortcut: str,
    ) -> None:
        super().__init__()
        self.layers = [
            _VaeResidualUnit(in_channels, dilation=1, use_snake=use_snake),
            _VaeResidualUnit(in_channels, dilation=3, use_snake=use_snake),
            _VaeResidualUnit(in_channels, dilation=9, use_snake=use_snake),
            _get_vae_activation(use_snake, in_channels),
            _ChannelFirstConv1d(
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=True,
            ),
        ]
        self.res = (
            _DownsampleShortcut(in_channels, out_channels, stride)
            if downsample_shortcut == "averaging"
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.res(x) if self.res is not None else None
        for layer in self.layers:
            x = layer(x)
        if residual is not None:
            x = x + residual
        return x


class _VaeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        use_snake: bool,
        upsample_shortcut: str,
    ) -> None:
        super().__init__()
        self.layers = [
            _get_vae_activation(use_snake, in_channels),
            _ChannelFirstConvTranspose1d(
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=True,
            ),
            _VaeResidualUnit(out_channels, dilation=1, use_snake=use_snake),
            _VaeResidualUnit(out_channels, dilation=3, use_snake=use_snake),
            _VaeResidualUnit(out_channels, dilation=9, use_snake=use_snake),
        ]
        self.res = (
            _UpsampleShortcut(in_channels, out_channels, stride)
            if upsample_shortcut == "duplicating"
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.res(x) if self.res is not None else None
        for layer in self.layers:
            x = layer(x)
        if residual is not None:
            x = x + residual
        return x


class AudioDiTVaeEncoder(nn.Module):
    def __init__(self, config: LongCatVaeConfig) -> None:
        super().__init__()
        c_mults = (1, *config.c_mults)
        channels = config.channels
        self.layers = [
            _ChannelFirstConv1d(
                config.in_channels, c_mults[0] * channels, 7, padding=3, bias=True
            )
        ]
        for index in range(len(c_mults) - 1):
            self.layers.append(
                _VaeEncoderBlock(
                    c_mults[index] * channels,
                    c_mults[index + 1] * channels,
                    config.strides[index],
                    use_snake=config.use_snake,
                    downsample_shortcut=config.downsample_shortcut,
                )
            )
        self.layers.append(
            _ChannelFirstConv1d(
                c_mults[-1] * channels,
                config.encoder_latent_dim,
                3,
                padding=1,
                bias=True,
            )
        )
        self.shortcut = (
            _DownsampleShortcut(c_mults[-1] * channels, config.encoder_latent_dim, 1)
            if config.out_shortcut == "averaging"
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.shortcut is None:
            for layer in self.layers:
                x = layer(x)
            return x

        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x) + self.shortcut(x)


class AudioDiTVaeDecoder(nn.Module):
    def __init__(self, config: LongCatVaeConfig) -> None:
        super().__init__()
        c_mults = (1, *config.c_mults)
        channels = config.channels
        self.shortcut = (
            _UpsampleShortcut(config.latent_dim, c_mults[-1] * channels, 1)
            if config.in_shortcut == "duplicating"
            else None
        )
        self.layers = [
            _ChannelFirstConv1d(
                config.latent_dim, c_mults[-1] * channels, 7, padding=3, bias=True
            )
        ]
        for index in range(len(c_mults) - 1, 0, -1):
            self.layers.append(
                _VaeDecoderBlock(
                    c_mults[index] * channels,
                    c_mults[index - 1] * channels,
                    config.strides[index - 1],
                    use_snake=config.use_snake,
                    upsample_shortcut=config.upsample_shortcut,
                )
            )
        self.layers.append(_get_vae_activation(config.use_snake, c_mults[0] * channels))
        self.layers.append(
            _ChannelFirstConv1d(
                c_mults[0] * channels, config.in_channels, 7, padding=3, bias=False
            )
        )
        self.layers.append(nn.Tanh() if config.final_tanh else _Identity())

    def __call__(self, x: mx.array) -> mx.array:
        if self.shortcut is None:
            for layer in self.layers:
                x = layer(x)
            return x

        x_short = self.shortcut(x) + self.layers[0](x)
        for layer in self.layers[1:]:
            x_short = layer(x_short)
        return x_short


class LongCatAudioDiTVae(nn.Module):
    def __init__(self, config: LongCatVaeConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AudioDiTVaeEncoder(config)
        self.decoder = AudioDiTVaeDecoder(config)
        self.scale = config.scale
        self.downsampling_ratio = config.downsampling_ratio

    def to_half(self) -> "LongCatAudioDiTVae":
        self.encoder.set_dtype(mx.float16)
        self.decoder.set_dtype(mx.float16)
        return self

    def encode(self, audio: mx.array) -> mx.array:
        audio = audio.astype(self.encoder.layers[0].weight.dtype)
        latents = self.encoder(audio)
        mean, scale_param = mx.split(latents, 2, axis=1)
        stdev = nn.softplus(scale_param) + 1e-4
        sampled = (mx.random.normal(mean.shape, dtype=mean.dtype) * stdev) + mean
        return sampled.astype(mx.float32) / self.scale

    def decode(self, latents: mx.array) -> mx.array:
        latents = (latents * self.scale).astype(self.decoder.layers[0].weight.dtype)
        decoded = self.decoder(latents)
        return decoded.astype(mx.float32)
