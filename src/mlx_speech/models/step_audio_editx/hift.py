"""HiFT vocoder support for Step-Audio-EditX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...checkpoints import load_torch_archive_state_dict
from ..step_audio_tokenizer.config import _parse_scalar
from ..step_audio_tokenizer.processor import _periodic_hann_window
from .frontend import resolve_step_audio_cosyvoice_dir


def _elu(x: mx.array) -> mx.array:
    return mx.where(x > 0, x, mx.exp(x) - 1.0)


def _leaky_relu(x: mx.array, negative_slope: float) -> mx.array:
    return mx.maximum(x, 0) + float(negative_slope) * mx.minimum(x, 0)


def _apply_conv1d(module: nn.Conv1d, x: mx.array) -> mx.array:
    return module(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def _apply_conv_transpose1d(module: nn.ConvTranspose1d, x: mx.array) -> mx.array:
    return module(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def _reflection_pad_left(x: mx.array, pad: int) -> mx.array:
    if pad <= 0:
        return x
    if int(x.shape[2]) <= 1:
        left = mx.repeat(x[:, :, :1], repeats=pad, axis=2)
        return mx.concatenate([left, x], axis=2)
    indices = list(range(1, pad + 1))
    left = x[:, :, indices[::-1]]
    return mx.concatenate([left, x], axis=2)


def _upsample_nearest_1d(x: mx.array, scale_factor: int) -> mx.array:
    return mx.repeat(x, repeats=int(scale_factor), axis=2)


def _linear_interpolate_time_align_corners_false(
    values: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Match torch.nn.functional.interpolate(..., mode='linear', align_corners=False)."""

    if target_length <= 0:
        raise ValueError(f"Target length must be positive, got {target_length}.")
    if values.ndim != 3:
        raise ValueError(f"Expected (batch, time, channels), got {values.shape}.")
    input_length = int(values.shape[1])
    if input_length == target_length:
        return values.astype(np.float32, copy=False)
    if input_length <= 0:
        raise ValueError("Input length must be positive for linear interpolation.")
    if input_length == 1:
        return np.repeat(values.astype(np.float32, copy=False), target_length, axis=1)

    scale = float(input_length) / float(target_length)
    source_positions = ((np.arange(target_length, dtype=np.float32) + 0.5) * scale) - 0.5
    source_positions = np.clip(source_positions, 0.0, float(input_length - 1))

    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, input_length - 1)
    lerp = (source_positions - left.astype(np.float32))[None, :, None]

    left_values = np.take(values, left, axis=1)
    right_values = np.take(values, right, axis=1)
    return ((1.0 - lerp) * left_values + lerp * right_values).astype(np.float32)


def _parse_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("!new:"):
        return None
    if stripped[0] in {"[", "{", "(", "'"} or stripped in {"True", "False"} or re.fullmatch(
        r"[+-]?\d+(\.\d+)?",
        stripped,
    ):
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            pass
    return _parse_scalar(stripped)


def _extract_nested_yaml_section(path: str | Path, section_name: str) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start_index: int | None = None
    section_indent = 0
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith(f"{section_name}:"):
            start_index = index + 1
            section_indent = len(raw_line) - len(raw_line.lstrip(" "))
            break
    if start_index is None:
        raise ValueError(f"Missing {section_name} in {path}")

    stack: list[tuple[int, dict[str, Any]]] = [(section_indent, {})]
    for raw_line in lines[start_index:]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent <= section_indent:
            break
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {section_name}: {stripped!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        parsed_value = _parse_value(value)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation in {path}: {stripped!r}")
        parent = stack[-1][1]
        if parsed_value is None and not value.strip():
            new_mapping: dict[str, Any] = {}
            parent[key] = new_mapping
            stack.append((indent, new_mapping))
        elif parsed_value is None and value.strip().startswith("!new:"):
            new_mapping = {}
            parent[key] = new_mapping
            stack.append((indent, new_mapping))
        else:
            parent[key] = parsed_value
    return stack[0][1]


def _materialize_weight_norm(v: mx.array, g: mx.array) -> mx.array:
    v32 = v.astype(mx.float32)
    g32 = g.astype(mx.float32)
    axes = tuple(range(1, v.ndim))
    norm = mx.sqrt(mx.sum(v32 * v32, axis=axes, keepdims=True))
    norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
    return (g32 * v32) / norm


def _torch_conv1d_to_mlx(weight: mx.array) -> mx.array:
    return weight.transpose(0, 2, 1)


def _torch_convtr1d_to_mlx(weight: mx.array) -> mx.array:
    return weight.transpose(1, 2, 0)


def _stft_real_imag(x: np.ndarray, *, n_fft: int, hop_len: int, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected waveform shape (batch, time), got {x.shape}.")
    pad = n_fft // 2
    padded = np.pad(x.astype(np.float32, copy=False), ((0, 0), (pad, pad)), mode="reflect")
    frame_count = 1 + (padded.shape[1] - n_fft) // hop_len
    frames = np.stack(
        [padded[:, start : start + n_fft] for start in range(0, frame_count * hop_len, hop_len)],
        axis=1,
    )
    windowed = frames * window[None, None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=-1)
    real = np.transpose(spectrum.real.astype(np.float32), (0, 2, 1))
    imag = np.transpose(spectrum.imag.astype(np.float32), (0, 2, 1))
    return real, imag


def _istft(magnitude: np.ndarray, phase: np.ndarray, *, n_fft: int, hop_len: int, window: np.ndarray) -> np.ndarray:
    complex_spec = magnitude * np.cos(phase) + 1j * (magnitude * np.sin(phase))
    frames = np.fft.irfft(np.transpose(complex_spec, (0, 2, 1)), n=n_fft, axis=-1).real
    out_len = hop_len * (frames.shape[1] - 1) + n_fft
    y = np.zeros((frames.shape[0], out_len), dtype=np.float32)
    window_sums = np.zeros((out_len,), dtype=np.float32)
    window_sq = (window.astype(np.float32) ** 2)
    for frame_index in range(frames.shape[1]):
        start = frame_index * hop_len
        y[:, start : start + n_fft] += frames[:, frame_index, :] * window[None, :]
        window_sums[start : start + n_fft] += window_sq
    nonzero = window_sums > 1e-8
    y[:, nonzero] /= window_sums[nonzero][None, :]
    pad = n_fft // 2
    if y.shape[1] > 2 * pad:
        y = y[:, pad:-pad]
    return y.astype(np.float32, copy=False)


@dataclass(frozen=True)
class StepAudioHiFTF0PredictorConfig:
    num_class: int
    in_channels: int
    cond_channels: int


@dataclass(frozen=True)
class StepAudioHiFTConfig:
    in_channels: int
    base_channels: int
    nb_harmonics: int
    sampling_rate: int
    nsf_alpha: float
    nsf_sigma: float
    nsf_voiced_threshold: float
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    istft_n_fft: int
    istft_hop_len: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    source_resblock_kernel_sizes: tuple[int, ...]
    source_resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    lrelu_slope: float
    audio_limit: float
    f0_predictor: StepAudioHiFTF0PredictorConfig

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "StepAudioHiFTConfig":
        payload = _extract_nested_yaml_section(path, "hift")
        f0_payload = payload.get("f0_predictor", {})
        if not isinstance(f0_payload, dict):
            raise ValueError("Expected f0_predictor mapping inside HiFT config.")
        istft_payload = payload.get("istft_params", {})
        if not isinstance(istft_payload, dict):
            raise ValueError("Expected istft_params mapping inside HiFT config.")
        return cls(
            in_channels=int(payload.get("in_channels", 80)),
            base_channels=int(payload.get("base_channels", 512)),
            nb_harmonics=int(payload.get("nb_harmonics", 8)),
            sampling_rate=int(payload.get("sampling_rate", 24000)),
            nsf_alpha=float(payload.get("nsf_alpha", 0.1)),
            nsf_sigma=float(payload.get("nsf_sigma", 0.003)),
            nsf_voiced_threshold=float(payload.get("nsf_voiced_threshold", 10.0)),
            upsample_rates=tuple(int(x) for x in payload.get("upsample_rates", [8, 5, 3])),
            upsample_kernel_sizes=tuple(int(x) for x in payload.get("upsample_kernel_sizes", [16, 11, 7])),
            istft_n_fft=int(istft_payload.get("n_fft", 16)),
            istft_hop_len=int(istft_payload.get("hop_len", 4)),
            resblock_kernel_sizes=tuple(int(x) for x in payload.get("resblock_kernel_sizes", [3, 7, 11])),
            resblock_dilation_sizes=tuple(
                tuple(int(v) for v in group) for group in payload.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
            ),
            source_resblock_kernel_sizes=tuple(
                int(x) for x in payload.get("source_resblock_kernel_sizes", [7, 7, 11])
            ),
            source_resblock_dilation_sizes=tuple(
                tuple(int(v) for v in group)
                for group in payload.get("source_resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
            ),
            lrelu_slope=float(payload.get("lrelu_slope", 0.1)),
            audio_limit=float(payload.get("audio_limit", 0.99)),
            f0_predictor=StepAudioHiFTF0PredictorConfig(
                num_class=int(f0_payload.get("num_class", 1)),
                in_channels=int(f0_payload.get("in_channels", 80)),
                cond_channels=int(f0_payload.get("cond_channels", 512)),
            ),
        )


@dataclass(frozen=True)
class StepAudioHiFTCheckpoint:
    model_dir: Path
    config: StepAudioHiFTConfig
    state_dict: dict[str, mx.array]


@dataclass(frozen=True)
class StepAudioHiFTAlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return (
            not self.missing_in_model
            and not self.missing_in_checkpoint
            and not self.shape_mismatches
        )


@dataclass(frozen=True)
class LoadedStepAudioHiFTModel:
    model_dir: Path
    config: StepAudioHiFTConfig
    checkpoint: StepAudioHiFTCheckpoint
    model: "StepAudioHiFTGenerator"
    alignment_report: StepAudioHiFTAlignmentReport


class StepAudioSnake(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,), dtype=mx.float32)
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha.reshape(1, -1, 1)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * mx.sin(x * alpha) ** 2


class StepAudioResBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=((kernel_size * dilation - dilation) // 2),
                dilation=dilation,
                bias=True,
            )
            for dilation in dilations
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                dilation=1,
                bias=True,
            )
            for _ in dilations
        ]
        self.activations1 = [StepAudioSnake(channels) for _ in dilations]
        self.activations2 = [StepAudioSnake(channels) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        for activation1, conv1, activation2, conv2 in zip(
            self.activations1,
            self.convs1,
            self.activations2,
            self.convs2,
        ):
            xt = activation1(x)
            xt = _apply_conv1d(conv1, xt)
            xt = activation2(xt)
            xt = _apply_conv1d(conv2, xt)
            x = xt + x
        return x


class StepAudioConvRNNF0Predictor(nn.Module):
    def __init__(self, config: StepAudioHiFTF0PredictorConfig):
        super().__init__()
        self.num_class = config.num_class
        self.condnet = [
            nn.Conv1d(
                config.in_channels if index == 0 else config.cond_channels,
                config.cond_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            for index in range(5)
        ]
        self.classifier = nn.Linear(config.cond_channels, config.num_class, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.condnet:
            x = _apply_conv1d(conv, x)
            x = _elu(x)
        x = x.transpose(0, 2, 1)
        return mx.abs(self.classifier(x).squeeze(-1))


class StepAudioSourceModuleHnNSF2(nn.Module):
    def __init__(
        self,
        *,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int,
        sine_amp: float,
        add_noise_std: float,
        voiced_threshold: float,
    ):
        super().__init__()
        self.sampling_rate = int(sampling_rate)
        self.upsample_scale = int(upsample_scale)
        self.harmonic_num = int(harmonic_num)
        self.sine_amp = float(sine_amp)
        self.noise_std = float(add_noise_std)
        self.voiced_threshold = float(voiced_threshold)
        self.l_linear = nn.Linear(self.harmonic_num + 1, 1, bias=True)
        self._rng = np.random.default_rng(seed=0)

    def _linear_resize_time(self, values: np.ndarray, target_length: int) -> np.ndarray:
        return _linear_interpolate_time_align_corners_false(values, target_length)

    def _f02uv(self, f0: np.ndarray) -> np.ndarray:
        return (f0 > self.voiced_threshold).astype(np.float32)

    def _f02sine(self, f0_values: np.ndarray) -> np.ndarray:
        rad_values = (f0_values / float(self.sampling_rate)) % 1.0
        rand_ini = self._rng.random((f0_values.shape[0], f0_values.shape[2]), dtype=np.float32)
        rand_ini[:, 0] = 0.0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        down_length = max(int(round(rad_values.shape[1] / float(self.upsample_scale))), 1)
        rad_values = self._linear_resize_time(rad_values, down_length)
        phase = np.cumsum(rad_values, axis=1) * (2.0 * np.pi)
        phase = self._linear_resize_time(phase, f0_values.shape[1]) * float(self.upsample_scale)
        return np.sin(phase).astype(np.float32)

    def __call__(self, f0: np.ndarray) -> tuple[mx.array, np.ndarray, np.ndarray]:
        fn = f0 * np.arange(1, self.harmonic_num + 2, dtype=np.float32).reshape(1, 1, -1)
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
        noise = noise_amp * self._rng.standard_normal(sine_waves.shape, dtype=np.float32)
        sine_waves = sine_waves * uv + noise
        sine_merge = mx.tanh(self.l_linear(mx.array(sine_waves, dtype=mx.float32)))
        return sine_merge, noise.astype(np.float32), uv.astype(np.float32)


class StepAudioHiFTGenerator(nn.Module):
    def __init__(self, config: StepAudioHiFTConfig):
        super().__init__()
        self.config = config
        self.out_channels = 1
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        upsample_scale = int(np.prod(config.upsample_rates) * config.istft_hop_len)
        self.m_source = StepAudioSourceModuleHnNSF2(
            sampling_rate=config.sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=config.nb_harmonics,
            sine_amp=config.nsf_alpha,
            add_noise_std=config.nsf_sigma,
            voiced_threshold=config.nsf_voiced_threshold,
        )
        self.f0_upsample_scale = upsample_scale
        self.conv_pre = nn.Conv1d(config.in_channels, config.base_channels, 7, stride=1, padding=3, bias=True)
        self.ups = [
            nn.ConvTranspose1d(
                config.base_channels // (2**index),
                config.base_channels // (2 ** (index + 1)),
                kernel_size,
                stride=rate,
                padding=(kernel_size - rate) // 2,
                bias=True,
            )
            for index, (rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]

        self.source_downs = []
        self.source_resblocks = []
        downsample_rates = [1] + list(config.upsample_rates[::-1][:-1])
        downsample_cum_rates = np.cumprod(downsample_rates)
        for index, (rate, kernel_size, dilations) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                config.source_resblock_kernel_sizes,
                config.source_resblock_dilation_sizes,
            )
        ):
            out_channels = config.base_channels // (2 ** (index + 1))
            if int(rate) == 1:
                conv = nn.Conv1d(config.istft_n_fft + 2, out_channels, 1, stride=1, padding=0, bias=True)
            else:
                conv = nn.Conv1d(
                    config.istft_n_fft + 2,
                    out_channels,
                    int(rate) * 2,
                    stride=int(rate),
                    padding=int(rate) // 2,
                    bias=True,
                )
            self.source_downs.append(conv)
            self.source_resblocks.append(
                StepAudioResBlock(
                    out_channels,
                    kernel_size=int(kernel_size),
                    dilations=tuple(int(v) for v in dilations),
                )
            )

        self.resblocks = []
        for index in range(len(self.ups)):
            channels = config.base_channels // (2 ** (index + 1))
            for kernel_size, dilations in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(
                    StepAudioResBlock(
                        channels,
                        kernel_size=int(kernel_size),
                        dilations=tuple(int(v) for v in dilations),
                    )
                )
        self.conv_post = nn.Conv1d(
            config.base_channels // (2 ** len(self.ups)),
            config.istft_n_fft + 2,
            7,
            stride=1,
            padding=3,
            bias=True,
        )
        self.f0_predictor = StepAudioConvRNNF0Predictor(config.f0_predictor)
        self._stft_window = _periodic_hann_window(config.istft_n_fft)

    def _stft(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _stft_real_imag(
            source,
            n_fft=self.config.istft_n_fft,
            hop_len=self.config.istft_hop_len,
            window=self._stft_window,
        )

    def _istft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        return _istft(
            magnitude,
            phase,
            n_fft=self.config.istft_n_fft,
            hop_len=self.config.istft_hop_len,
            window=self._stft_window,
        )

    def decode_without_stft(self, x: mx.array, s_stft: mx.array) -> mx.array:
        x = _apply_conv1d(self.conv_pre, x)
        for index in range(self.num_upsamples):
            x = _leaky_relu(x, self.config.lrelu_slope)
            x = _apply_conv_transpose1d(self.ups[index], x)
            if index == self.num_upsamples - 1:
                x = _reflection_pad_left(x, 1)
            si = _apply_conv1d(self.source_downs[index], s_stft)
            si = self.source_resblocks[index](si)
            x = x + si

            xs = None
            for block_index in range(self.num_kernels):
                block = self.resblocks[index * self.num_kernels + block_index]
                block_out = block(x)
                xs = block_out if xs is None else xs + block_out
            x = xs / float(self.num_kernels)

        # Torch reference uses the default F.leaky_relu slope here, not the HiFT config slope.
        x = _leaky_relu(x, 0.01)
        x = _apply_conv1d(self.conv_post, x)
        return x

    def decode(self, x: mx.array, source_wave: mx.array) -> np.ndarray:
        source_np = np.asarray(source_wave, dtype=np.float32)
        if source_np.ndim != 3 or source_np.shape[1] != 1:
            raise ValueError(f"Expected source waveform shape (batch, 1, time), got {source_np.shape}.")
        s_stft_real, s_stft_imag = self._stft(source_np[:, 0, :])
        s_stft = mx.array(np.concatenate([s_stft_real, s_stft_imag], axis=1), dtype=mx.float32)
        decoded = self.decode_without_stft(x, s_stft)
        decoded_np = np.asarray(decoded, dtype=np.float32)
        freq_bins = self.config.istft_n_fft // 2 + 1
        magnitude = np.exp(decoded_np[:, :freq_bins, :].astype(np.float32))
        magnitude = np.minimum(magnitude, np.float32(1e2)).astype(np.float32, copy=False)
        phase = np.sin(decoded_np[:, freq_bins:, :].astype(np.float32))
        waveform = self._istft(magnitude, phase)
        return np.clip(waveform, -self.config.audio_limit, self.config.audio_limit).astype(np.float32)

    def inference(self, speech_feat: np.ndarray, cache_source: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        mel = np.asarray(speech_feat, dtype=np.float32)
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape (batch, channels, time), got {mel.shape}.")
        mel_mx = mx.array(mel, dtype=mx.float32)
        f0 = self.f0_predictor(mel_mx)
        f0_up = _upsample_nearest_1d(f0[:, None, :], self.f0_upsample_scale).transpose(0, 2, 1)
        source_merge, _, _ = self.m_source(np.asarray(f0_up, dtype=np.float32))
        source_wave = np.asarray(source_merge, dtype=np.float32).transpose(0, 2, 1)
        if cache_source is not None and int(np.asarray(cache_source).shape[2]) != 0:
            source_wave[:, :, : cache_source.shape[2]] = cache_source
        waveform = self.decode(mel_mx, mx.array(source_wave, dtype=mx.float32))
        return waveform, source_wave.astype(np.float32)


def sanitize_step_audio_hift_state_dict(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    materialized: dict[str, mx.array] = {}

    for key, value in weights.items():
        if ".parametrizations.weight.original0" in key:
            continue
        if ".parametrizations.weight.original1" in key:
            prefix = key.replace(".parametrizations.weight.original1", "")
            g_key = f"{prefix}.parametrizations.weight.original0"
            weight = _materialize_weight_norm(value, weights[g_key])
            if prefix.startswith("ups."):
                materialized[f"{prefix}.weight"] = _torch_convtr1d_to_mlx(weight)
            else:
                materialized[f"{prefix}.weight"] = _torch_conv1d_to_mlx(weight)
            continue
        if key.endswith(".weight") and value.ndim == 3:
            materialized[key] = _torch_conv1d_to_mlx(value)
            continue
        materialized[key] = value

    remapped: dict[str, mx.array] = {}
    for key, value in materialized.items():
        if key.startswith("f0_predictor.condnet."):
            match = re.match(r"f0_predictor\.condnet\.(\d+)\.(weight|bias)$", key)
            if match is not None:
                layer_index = int(match.group(1)) // 2
                key = f"f0_predictor.condnet.{layer_index}.{match.group(2)}"
        remapped[key] = value
    return remapped


def load_step_audio_hift_checkpoint(model_dir: str | Path) -> StepAudioHiFTCheckpoint:
    resolved_model_dir = resolve_step_audio_cosyvoice_dir(model_dir)
    config = StepAudioHiFTConfig.from_yaml_path(resolved_model_dir / "cosyvoice.yaml")
    archive = load_torch_archive_state_dict(resolved_model_dir / "hift.pt")
    state_dict = sanitize_step_audio_hift_state_dict(archive.weights)
    return StepAudioHiFTCheckpoint(
        model_dir=resolved_model_dir,
        config=config,
        state_dict=state_dict,
    )


def validate_step_audio_hift_checkpoint_against_model(
    model: StepAudioHiFTGenerator,
    checkpoint: StepAudioHiFTCheckpoint,
) -> StepAudioHiFTAlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioHiFTAlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_step_audio_hift_model(
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> LoadedStepAudioHiFTModel:
    checkpoint = load_step_audio_hift_checkpoint(model_dir)
    model = StepAudioHiFTGenerator(checkpoint.config)
    report = validate_step_audio_hift_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio HiFT checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return LoadedStepAudioHiFTModel(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
    )


__all__ = [
    "LoadedStepAudioHiFTModel",
    "StepAudioHiFTAlignmentReport",
    "StepAudioHiFTCheckpoint",
    "StepAudioHiFTConfig",
    "StepAudioHiFTF0PredictorConfig",
    "StepAudioHiFTGenerator",
    "load_step_audio_hift_checkpoint",
    "load_step_audio_hift_model",
    "sanitize_step_audio_hift_state_dict",
    "validate_step_audio_hift_checkpoint_against_model",
]
