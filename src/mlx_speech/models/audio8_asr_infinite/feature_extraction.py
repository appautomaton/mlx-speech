"""Voxtral-compatible log-Mel extraction for Audio8."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel(frequency: float) -> float:
    if frequency < _MIN_LOG_HZ:
        return frequency / _F_SP
    return _MIN_LOG_MEL + math.log(frequency / _MIN_LOG_HZ) / _LOGSTEP


def _mel_to_hz(mel: float) -> float:
    if mel < _MIN_LOG_MEL:
        return _F_SP * mel
    return _MIN_LOG_HZ * math.exp(_LOGSTEP * (mel - _MIN_LOG_MEL))


def _slaney_mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    frequencies = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, dtype=np.float64)
    mel_points = np.linspace(
        _hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2, dtype=np.float64
    )
    hz_points = np.asarray([_mel_to_hz(value) for value in mel_points])
    ramps = hz_points[:, None] - frequencies[None, :]
    filters = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / (hz_points[index + 1] - hz_points[index])
        upper = ramps[index + 2] / (hz_points[index + 2] - hz_points[index + 1])
        filters[index] = np.maximum(0.0, np.minimum(lower, upper))
    filters *= (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]))[:, None]
    return filters.astype(np.float32)


@dataclass(frozen=True)
class Audio8FeatureConfig:
    feature_size: int = 128
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    win_length: int = 400
    global_log_mel_max: float | None = 1.5

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Audio8FeatureConfig":
        path = Path(model_dir) / "preprocessor_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Audio8 preprocessor config not found: {path}")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            feature_size=int(payload["feature_size"]),
            sampling_rate=int(payload["sampling_rate"]),
            hop_length=int(payload["hop_length"]),
            n_fft=int(payload["n_fft"]),
            win_length=int(payload["win_length"]),
            global_log_mel_max=(
                float(payload["global_log_mel_max"])
                if payload.get("global_log_mel_max") is not None
                else None
            ),
        )


class Audio8FeatureExtractor:
    def __init__(self, config: Audio8FeatureConfig) -> None:
        if config.win_length != config.n_fft:
            raise ValueError("Audio8 requires win_length == n_fft")
        self.config = config
        self.mel_filters = mx.array(
            _slaney_mel_filters(config.sampling_rate, config.n_fft, config.feature_size)
        )
        self.window = mx.array(np.hanning(config.n_fft + 1)[:-1].astype(np.float32))

    def __call__(
        self, waveform: mx.array | np.ndarray, *, center: bool = True
    ) -> mx.array:
        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim != 1:
            raise ValueError(f"Expected mono PCM [samples], got {wave.shape}")
        config = self.config
        if wave.shape[0] < config.n_fft:
            raise ValueError(f"Audio8 requires at least {config.n_fft} samples per window")
        if center:
            wave = np.pad(wave, (config.n_fft // 2, config.n_fft // 2), mode="reflect")
        frame_count = 1 + (wave.shape[0] - config.n_fft) // config.hop_length
        frames = mx.as_strided(
            mx.array(wave),
            shape=(frame_count, config.n_fft),
            strides=(config.hop_length, 1),
        )
        spectrum = mx.fft.rfft(frames * self.window[None, :], axis=-1)
        power = mx.square(mx.abs(spectrum[:-1, :])).astype(mx.float32)
        mel = self.mel_filters @ power.T
        log_spec = mx.log(mx.maximum(mel, 1e-10)) / math.log(10.0)
        maximum = (
            mx.array(config.global_log_mel_max, dtype=mx.float32)
            if config.global_log_mel_max is not None
            else mx.max(log_spec)
        )
        log_spec = mx.maximum(log_spec, maximum - 8.0)
        return ((log_spec + 4.0) / 4.0)[None, :, :].astype(mx.float32)
