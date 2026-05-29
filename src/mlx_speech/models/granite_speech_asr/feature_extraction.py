"""Granite Speech log-mel feature extraction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GraniteSpeechConfig


@dataclass(frozen=True)
class GraniteSpeechAudioShape:
    """Preflight sizing for Granite audio features."""

    sample_count: int
    mel_frames: int
    encoder_frames: int
    audio_tokens: int


def _periodic_hann(size: int) -> np.ndarray:
    return np.array(
        [0.5 * (1.0 - math.cos(2.0 * math.pi * n / size)) for n in range(size)],
        dtype=np.float32,
    )


def _htk_hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _htk_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)


_MEL_FILTER_CACHE: dict[tuple[int, int, int, float, float], np.ndarray] = {}


def _htk_mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """HTK triangular mel filters matching mlx_audio.dsp.mel_filters defaults."""
    f_max = float(f_max if f_max is not None else sample_rate / 2)
    key = (sample_rate, n_fft, n_mels, float(f_min), f_max)
    if key in _MEL_FILTER_CACHE:
        return _MEL_FILTER_CACHE[key]

    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0.0, sample_rate / 2, n_freqs, dtype=np.float64)
    mel_min = _htk_hz_to_mel(float(f_min))
    mel_max = _htk_hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = _htk_mel_to_hz(mel_points)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = (all_freqs - hz_points[i]) / (hz_points[i + 1] - hz_points[i])
        upper = (hz_points[i + 2] - all_freqs) / (hz_points[i + 2] - hz_points[i + 1])
        filters[i] = np.maximum(0.0, np.minimum(lower, upper))

    result = filters.astype(np.float32)
    _MEL_FILTER_CACHE[key] = result
    return result


def _stft_power(
    waveform: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """Reflect-centered STFT power with frames on axis 0."""
    if waveform.size == 0:
        raise ValueError("Expected non-empty waveform")

    pad = n_fft // 2
    if waveform.size == 1:
        padded = np.pad(waveform, (pad, pad), mode="edge")
    else:
        padded = np.pad(waveform, (pad, pad), mode="reflect")

    window = _periodic_hann(win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = np.pad(window, (pad_left, pad_right))

    n_frames = 1 + (len(padded) - n_fft) // hop_length
    n_freqs = n_fft // 2 + 1
    power = np.zeros((n_frames, n_freqs), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power[i] = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    return power


class GraniteSpeechFeatureExtractor:
    """Pure-numpy audio frontend for Granite Speech."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        window_size: int = 15,
        downsample_rate: int = 5,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_size = window_size
        self.downsample_rate = downsample_rate
        _htk_mel_filters(sample_rate, n_fft, n_mels)

    @classmethod
    def from_config(cls, config: GraniteSpeechConfig) -> "GraniteSpeechFeatureExtractor":
        return cls(
            n_mels=config.encoder.input_dim // 2,
            window_size=config.window_size,
            downsample_rate=config.downsample_rate,
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "GraniteSpeechFeatureExtractor":
        model_dir = Path(model_dir)
        config = GraniteSpeechConfig.from_path(model_dir)
        preprocessor_path = model_dir / "preprocessor_config.json"
        if not preprocessor_path.exists():
            return cls.from_config(config)
        with preprocessor_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        mel = payload.get("melspec_kwargs", {})
        return cls(
            sample_rate=int(mel.get("sample_rate", payload.get("sampling_rate", 16000))),
            n_fft=int(mel.get("n_fft", 512)),
            win_length=int(mel.get("win_length", 400)),
            hop_length=int(mel.get("hop_length", 160)),
            n_mels=int(mel.get("n_mels", config.encoder.input_dim // 2)),
            window_size=int(payload.get("projector_window_size", config.window_size)),
            downsample_rate=int(payload.get("projector_downsample_rate", config.downsample_rate)),
        )

    def preflight_shape(self, sample_count: int) -> GraniteSpeechAudioShape:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        mel_frames = 1 + sample_count // self.hop_length
        encoder_frames = mel_frames // 2
        audio_tokens = math.ceil(encoder_frames / self.window_size) * (
            self.window_size // self.downsample_rate
        )
        return GraniteSpeechAudioShape(
            sample_count=sample_count,
            mel_frames=mel_frames,
            encoder_frames=encoder_frames,
            audio_tokens=audio_tokens,
        )

    def __call__(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D mono waveform, got shape {waveform.shape}")
        if waveform.size == 0:
            raise ValueError("Expected non-empty waveform")

        power = _stft_power(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel_filters = _htk_mel_filters(self.sample_rate, self.n_fft, self.n_mels)
        mel_spec = power @ mel_filters.T

        logmel = np.log10(np.clip(mel_spec, 1e-10, None))
        max_logmel = float(np.max(logmel))
        logmel = np.maximum(logmel, max_logmel - 8.0) / 4.0 + 1.0

        if logmel.shape[0] % 2 == 1:
            logmel = logmel[:-1]

        encoder_input = logmel.reshape(-1, 2 * self.n_mels).astype(np.float32)
        shape = self.preflight_shape(waveform.shape[0])
        return encoder_input[None, :, :], shape.audio_tokens
