"""Pure-MLX CAM++ speaker conditioning for dots.tts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


SPEAKER_SAMPLE_RATE = 16_000
SPEAKER_FEATURES = 80


def _sinc_resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Match torchaudio's default Hann-windowed sinc resampler."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("speaker sample rates must be positive")
    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or value.size == 0:
        return value
    divisor = math.gcd(source_rate, target_rate)
    source = source_rate // divisor
    target = target_rate // divisor
    lowpass_width = 6
    rolloff = 0.99
    base_frequency = min(source, target) * rolloff
    width = math.ceil(lowpass_width * source / base_frequency)
    index = np.arange(-width, width + source, dtype=np.float64) / source
    phase = np.arange(0, -target, -1, dtype=np.float64)[:, None, None] / target
    time = (phase + index[None, None, :]) * base_frequency
    time = np.clip(time, -lowpass_width, lowpass_width)
    window = np.cos(time * np.pi / lowpass_width / 2.0) ** 2
    sinc = np.sinc(time)
    kernels = (sinc * window * (base_frequency / source)).astype(np.float32)

    padded = np.pad(value, (width, width + source))
    phases = []
    for kernel in kernels[:, 0, :]:
        phases.append(np.correlate(padded, kernel, mode="valid")[::source])
    output = np.stack(phases, axis=1).reshape(-1)
    target_length = math.ceil(target * value.size / source)
    return output[:target_length].astype(np.float32, copy=False)


def _next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _mel_scale(frequency: np.ndarray | float) -> np.ndarray:
    return 1127.0 * np.log1p(np.asarray(frequency) / 700.0)


def _mel_banks(
    bins: int, padded_window: int, sample_rate: int
) -> np.ndarray:
    fft_bins = padded_window // 2
    bin_width = sample_rate / padded_window
    low = float(_mel_scale(20.0))
    high = float(_mel_scale(sample_rate / 2.0))
    delta = (high - low) / (bins + 1)
    mel = _mel_scale(bin_width * np.arange(fft_bins, dtype=np.float64))[None]
    indices = np.arange(bins, dtype=np.float64)[:, None]
    left = low + indices * delta
    center = left + delta
    right = center + delta
    return np.maximum(
        0.0,
        np.minimum((mel - left) / (center - left), (right - mel) / (right - center)),
    )


def kaldi_fbank(waveform: np.ndarray, *, sample_rate: int = SPEAKER_SAMPLE_RATE) -> np.ndarray:
    """Compute the dots.tts Kaldi-compatible 80-bin fbank with mean normalization."""
    if sample_rate != SPEAKER_SAMPLE_RATE:
        raise ValueError(f"speaker fbank requires {SPEAKER_SAMPLE_RATE} Hz audio")
    value = np.asarray(waveform, dtype=np.float64).reshape(-1)
    window_size = int(sample_rate * 0.025)
    window_shift = int(sample_rate * 0.010)
    if value.size < window_size:
        raise ValueError("speaker audio is shorter than one 25 ms fbank frame")
    frame_count = 1 + (value.size - window_size) // window_shift
    indices = (
        np.arange(window_size)[None]
        + window_shift * np.arange(frame_count)[:, None]
    )
    frames = value[indices]
    frames -= frames.mean(axis=1, keepdims=True)
    previous = np.concatenate((frames[:, :1], frames[:, :-1]), axis=1)
    frames -= 0.97 * previous
    position = np.arange(window_size)
    povey = (0.5 - 0.5 * np.cos(2.0 * np.pi * position / (window_size - 1))) ** 0.85
    frames *= povey[None]
    padded_window = _next_power_of_two(window_size)
    frames = np.pad(frames, ((0, 0), (0, padded_window - window_size)))
    spectrum = np.abs(np.fft.rfft(frames, n=padded_window, axis=1)) ** 2
    filters = np.pad(
        _mel_banks(SPEAKER_FEATURES, padded_window, sample_rate),
        ((0, 0), (0, 1)),
    )
    energies = spectrum @ filters.T
    features = np.log(np.maximum(energies, np.finfo(np.float32).eps))
    features -= features.mean(axis=0, keepdims=True)
    return features.astype(np.float32)


class SpeakerFrontend:
    def __init__(self, *, max_audio_seconds: float = 10.0):
        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        self.max_audio_seconds = float(max_audio_seconds)

    @staticmethod
    def _mono(audio: np.ndarray | mx.array) -> np.ndarray:
        value = np.asarray(audio, dtype=np.float32)
        if value.ndim == 1:
            return value
        if value.ndim != 2:
            raise ValueError(f"speaker audio must be mono or stereo, got {value.shape}")
        if value.shape[0] <= 8 and value.shape[0] < value.shape[1]:
            return value.mean(axis=0, dtype=np.float32)
        return value.mean(axis=1, dtype=np.float32)

    def features(
        self, audio: np.ndarray | mx.array, *, sample_rate: int
    ) -> tuple[np.ndarray, int]:
        waveform = self._mono(audio)
        if not np.isfinite(waveform).all():
            raise ValueError("speaker audio contains non-finite values")
        maximum = round(sample_rate * self.max_audio_seconds)
        waveform = waveform[:maximum]
        waveform = _sinc_resample(waveform, sample_rate, SPEAKER_SAMPLE_RATE)
        features = kaldi_fbank(waveform)
        return features, int(features.shape[0])


class FrozenBatchNorm(nn.Module):
    """Evaluation-only BatchNorm with explicit checkpoint buffers."""

    def __init__(self, channels: int, *, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((channels,)) if affine else None
        self.bias = mx.zeros((channels,)) if affine else None
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        shape = (1,) * (value.ndim - 1) + (-1,)
        normalized = (value.astype(mx.float32) - self.running_mean.reshape(shape)) * mx.rsqrt(
            self.running_var.reshape(shape) + self.eps
        )
        if self.weight is not None:
            normalized *= self.weight.reshape(shape)
        if self.bias is not None:
            normalized += self.bias.reshape(shape)
        return normalized.astype(value.dtype)


class _Conv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = mx.random.normal((output_channels, kernel_size, input_channels)) * 0.02
        self.bias = mx.zeros((output_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, value: mx.array) -> mx.array:
        output = mx.conv1d(
            value,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return output if self.bias is None else output + self.bias


class _Conv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        bias: bool = False,
    ):
        super().__init__()
        self.weight = mx.random.normal(
            (output_channels, kernel_size[0], kernel_size[1], input_channels)
        ) * 0.02
        self.bias = mx.zeros((output_channels,)) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, value: mx.array) -> mx.array:
        output = mx.conv2d(
            value, self.weight, stride=self.stride, padding=self.padding
        )
        return output if self.bias is None else output + self.bias


class _ResBlock(nn.Module):
    def __init__(self, channels: int, *, stride: int):
        super().__init__()
        self.conv1 = _Conv2d(
            channels, channels, (3, 3), stride=(stride, 1), padding=(1, 1)
        )
        self.bn1 = FrozenBatchNorm(channels)
        self.conv2 = _Conv2d(channels, channels, (3, 3), padding=(1, 1))
        self.bn2 = FrozenBatchNorm(channels)
        self.shortcut = (
            _Conv2d(channels, channels, (1, 1), stride=(stride, 1))
            if stride != 1
            else None
        )
        self.shortcut_bn = FrozenBatchNorm(channels) if self.shortcut is not None else None

    def __call__(self, value: mx.array) -> mx.array:
        output = nn.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        shortcut = value
        if self.shortcut is not None:
            shortcut = self.shortcut_bn(self.shortcut(value))
        return nn.relu(output + shortcut)


class _FCM(nn.Module):
    def __init__(self, feature_dim: int, channels: int):
        super().__init__()
        if feature_dim % 8:
            raise ValueError("CAM++ feature dimension must be divisible by 8")
        self.conv1 = _Conv2d(1, channels, (3, 3), padding=(1, 1))
        self.bn1 = FrozenBatchNorm(channels)
        self.layer1 = [_ResBlock(channels, stride=2), _ResBlock(channels, stride=1)]
        self.layer2 = [_ResBlock(channels, stride=2), _ResBlock(channels, stride=1)]
        self.conv2 = _Conv2d(
            channels, channels, (3, 3), stride=(2, 1), padding=(1, 1)
        )
        self.bn2 = FrozenBatchNorm(channels)
        self.output_channels = channels * (feature_dim // 8)

    def __call__(self, value: mx.array) -> mx.array:
        value = value.transpose(0, 2, 1)[..., None]
        output = nn.relu(self.bn1(self.conv1(value)))
        for block in (*self.layer1, *self.layer2):
            output = block(output)
        output = nn.relu(self.bn2(self.conv2(output)))
        batch, height, time, channels = output.shape
        return output.transpose(0, 2, 3, 1).reshape(
            batch, time, channels * height
        )


def _segment_pool(value: mx.array, length: int = 100) -> mx.array:
    batch, time, channels = value.shape
    segments = []
    for start in range(0, int(time), length):
        end = min(start + length, int(time))
        mean = value[:, start:end].mean(axis=1, keepdims=True)
        segments.append(mx.repeat(mean, end - start, axis=1))
    return mx.concatenate(segments, axis=1).reshape(batch, time, channels)


class _CAMLayer(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, *, dilation: int
    ):
        super().__init__()
        self.linear_local = _Conv1d(
            input_channels,
            output_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.linear1 = _Conv1d(
            input_channels, input_channels // 2, 1, bias=True
        )
        self.linear2 = _Conv1d(
            input_channels // 2, output_channels, 1, bias=True
        )

    def __call__(self, value: mx.array) -> mx.array:
        local = self.linear_local(value)
        context = value.mean(axis=1, keepdims=True) + _segment_pool(value)
        gate = mx.sigmoid(self.linear2(nn.relu(self.linear1(context))))
        return local * gate


class _DenseTDNNLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        *,
        dilation: int,
    ):
        super().__init__()
        self.nonlinear1 = FrozenBatchNorm(input_channels)
        self.linear1 = _Conv1d(input_channels, bottleneck_channels, 1)
        self.nonlinear2 = FrozenBatchNorm(bottleneck_channels)
        self.cam_layer = _CAMLayer(
            bottleneck_channels, output_channels, dilation=dilation
        )

    def __call__(self, value: mx.array) -> mx.array:
        value = self.linear1(nn.relu(self.nonlinear1(value)))
        return self.cam_layer(nn.relu(self.nonlinear2(value)))


class _DenseBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        input_channels: int,
        growth_rate: int,
        bottleneck_channels: int,
        *,
        dilation: int,
    ):
        super().__init__()
        self.layers = [
            _DenseTDNNLayer(
                input_channels + index * growth_rate,
                growth_rate,
                bottleneck_channels,
                dilation=dilation,
            )
            for index in range(layers)
        ]

    def __call__(self, value: mx.array) -> mx.array:
        for layer in self.layers:
            value = mx.concatenate((value, layer(value)), axis=-1)
        return value


class _Transit(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.nonlinear = FrozenBatchNorm(input_channels)
        self.linear = _Conv1d(input_channels, output_channels, 1)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear(nn.relu(self.nonlinear(value)))


class _TDNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.linear = _Conv1d(
            input_channels, output_channels, 5, stride=2, padding=2
        )
        self.nonlinear = FrozenBatchNorm(output_channels)

    def __call__(self, value: mx.array) -> mx.array:
        return nn.relu(self.nonlinear(self.linear(value)))


@dataclass(frozen=True)
class CAMPPlusConfig:
    feature_dim: int = 80
    embedding_size: int = 512
    growth_rate: int = 32
    bottleneck_size: int = 4
    initial_channels: int = 128
    block_layers: tuple[int, int, int] = (12, 24, 16)


class CAMPPlus(nn.Module):
    def __init__(self, config: CAMPPlusConfig = CAMPPlusConfig()):
        super().__init__()
        self.config = config
        self.head = _FCM(config.feature_dim, config.growth_rate)
        channels = self.head.output_channels
        self.tdnn = _TDNN(channels, config.initial_channels)
        channels = config.initial_channels
        self.blocks = []
        self.transits = []
        for layers, dilation in zip(config.block_layers, (1, 2, 2), strict=True):
            block = _DenseBlock(
                layers,
                channels,
                config.growth_rate,
                config.bottleneck_size * config.growth_rate,
                dilation=dilation,
            )
            self.blocks.append(block)
            channels += layers * config.growth_rate
            transit = _Transit(channels, channels // 2)
            self.transits.append(transit)
            channels //= 2
        self.out_nonlinear = FrozenBatchNorm(channels)
        self.dense = _Conv1d(channels * 2, config.embedding_size, 1)
        self.dense_norm = FrozenBatchNorm(config.embedding_size, affine=False)

    @staticmethod
    def _masked_statistics(
        value: mx.array, lengths: mx.array, *, eps: float = 1e-2
    ) -> mx.array:
        time = int(value.shape[1])
        lengths = mx.clip(lengths.astype(mx.int32), 1, time)
        mask = mx.arange(time, dtype=mx.int32)[None, :] < lengths[:, None]
        mask = mask.astype(value.dtype)[..., None]
        denominator = lengths.astype(value.dtype)[:, None]
        mean = mx.sum(value * mask, axis=1) / denominator
        centered = (value - mean[:, None]) * mask
        variance_denominator = mx.maximum(lengths - 1, 1).astype(value.dtype)[:, None]
        variance = mx.sum(centered * centered, axis=1) / variance_denominator
        standard_deviation = mx.sqrt(mx.maximum(variance, eps))
        return mx.concatenate((mean, standard_deviation), axis=-1)

    def __call__(
        self, features: mx.array, *, lengths: mx.array | None = None
    ) -> mx.array:
        if features.ndim != 3 or int(features.shape[-1]) != self.config.feature_dim:
            raise ValueError(
                f"CAM++ expects (batch, time, {self.config.feature_dim}), "
                f"got {features.shape}"
            )
        value = self.tdnn(self.head(features))
        if lengths is not None:
            lengths = (
                mx.floor_divide(lengths.astype(mx.int32) - 1, 2) + 1
            )
        for block, transit in zip(self.blocks, self.transits, strict=True):
            value = transit(block(value))
        value = nn.relu(self.out_nonlinear(value))
        if lengths is None:
            mean = value.mean(axis=1)
            centered = value - mean[:, None]
            denominator = max(int(value.shape[1]) - 1, 1)
            standard_deviation = mx.sqrt(
                mx.sum(centered * centered, axis=1) / denominator
            )
            pooled = mx.concatenate((mean, standard_deviation), axis=-1)
        else:
            pooled = self._masked_statistics(value, lengths)
        pooled = pooled[:, None]
        return self.dense_norm(self.dense(pooled))[:, 0]


@dataclass(frozen=True)
class SpeakerConditioning:
    features: mx.array
    embedding: mx.array
    scaled_embedding: mx.array
    projected: mx.array


class SpeakerConditioner(nn.Module):
    def __init__(
        self,
        *,
        encoder: CAMPPlus | None = None,
        conditioning_dim: int = 1024,
        max_audio_seconds: float = 10.0,
    ):
        super().__init__()
        self.frontend = SpeakerFrontend(max_audio_seconds=max_audio_seconds)
        self.encoder = CAMPPlus() if encoder is None else encoder
        self.projection = nn.Linear(
            self.encoder.config.embedding_size, conditioning_dim, bias=True
        )
        self.projection_norm = nn.LayerNorm(conditioning_dim)

    def __call__(
        self,
        audio: np.ndarray | mx.array,
        *,
        sample_rate: int,
        speaker_scale: float = 1.5,
    ) -> SpeakerConditioning:
        if not np.isfinite(speaker_scale):
            raise ValueError("speaker_scale must be finite")
        features, length = self.frontend.features(audio, sample_rate=sample_rate)
        feature_array = mx.array(features[None], dtype=mx.float32)
        embedding = self.encoder(
            feature_array, lengths=mx.array([length], dtype=mx.int32)
        )
        scaled_embedding = embedding * float(speaker_scale)
        projected = self.projection_norm(
            self.projection(scaled_embedding)
        )
        return SpeakerConditioning(
            features=feature_array,
            embedding=embedding,
            scaled_embedding=scaled_embedding,
            projected=projected,
        )


__all__ = [
    "CAMPPlus",
    "CAMPPlusConfig",
    "FrozenBatchNorm",
    "SPEAKER_FEATURES",
    "SPEAKER_SAMPLE_RATE",
    "SpeakerConditioner",
    "SpeakerConditioning",
    "SpeakerFrontend",
    "kaldi_fbank",
]
