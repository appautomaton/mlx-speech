"""Nemotron 3.5 ASR log-mel feature extraction in MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import PreprocessArgs

_SLANEY_F_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP
_SLANEY_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel(freq: float) -> float:
    if freq < _SLANEY_MIN_LOG_HZ:
        return freq / _SLANEY_F_SP
    return _SLANEY_MIN_LOG_MEL + math.log(freq / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP


def _mel_to_hz(mel: float) -> float:
    if mel < _SLANEY_MIN_LOG_MEL:
        return mel * _SLANEY_F_SP
    return _SLANEY_MIN_LOG_HZ * math.exp(_SLANEY_LOGSTEP * (mel - _SLANEY_MIN_LOG_MEL))


def _slaney_mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    *,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Match ``librosa.filters.mel(..., norm="slaney")`` used by NeMo."""
    upper = float(sample_rate / 2 if f_max is None else f_max)
    frequencies = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, dtype=np.float64)
    mel_points = np.linspace(
        _hz_to_mel(float(f_min)),
        _hz_to_mel(upper),
        n_mels + 2,
        dtype=np.float64,
    )
    hz_points = np.asarray([_mel_to_hz(value) for value in mel_points])

    ramps = hz_points[:, None] - frequencies[None, :]
    filters = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / (hz_points[index + 1] - hz_points[index])
        upper_slope = ramps[index + 2] / (hz_points[index + 2] - hz_points[index + 1])
        filters[index] = np.maximum(0.0, np.minimum(lower, upper_slope))

    area_norm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filters *= area_norm[:, None]
    return filters.astype(np.float32)


class NemotronFeatureExtractor(nn.Module):
    """Inference-time NeMo ``AudioToMelSpectrogramPreprocessor`` parity.

    NeMo stores ``dither=1e-5`` in the checkpoint config but applies it only
    while the module is in training mode. This runtime is inference-only, so
    feature extraction is deterministic and does not add dither.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        preemphasis: float = 0.97,
        dither: float = 1e-5,
        normalize: str = "NA",
        log_zero_guard_value: float = 2**-24,
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        if min(sample_rate, n_fft, win_length, hop_length, n_mels) <= 0:
            raise ValueError("feature-extractor dimensions must be positive")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if normalize != "NA":
            raise NotImplementedError("Nemotron runtime requires normalize='NA'")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.preemphasis = preemphasis
        self.dither = dither
        self.normalize = normalize
        self.log_zero_guard_value = log_zero_guard_value
        self.pad_value = pad_value

        filters = _slaney_mel_filters(sample_rate, n_fft, n_mels)
        # Public names and shapes match the two NeMo featurizer buffers.
        self.fb = mx.array(filters[None, :, :])
        self.window = mx.array(np.hanning(win_length).astype(np.float32))

    def __call__(self, waveform: mx.array | np.ndarray) -> tuple[mx.array, mx.array]:
        """Return ``(features, lengths)`` as ``[1, T, 128]`` and ``[1]``."""
        wave = mx.array(waveform, dtype=mx.float32)
        if wave.ndim != 1:
            raise ValueError(f"expected a 1D mono waveform, got shape {wave.shape}")
        if wave.shape[0] == 0:
            raise ValueError("expected a non-empty waveform")

        if self.preemphasis > 0.0:
            wave = mx.concatenate(
                [wave[:1], wave[1:] - self.preemphasis * wave[:-1]], axis=0
            )

        center = self.n_fft // 2
        padded = mx.pad(wave, ((center, center),), constant_values=0.0)
        frame_count = 1 + (padded.shape[0] - self.n_fft) // self.hop_length
        frames = mx.as_strided(
            padded,
            shape=(frame_count, self.n_fft),
            strides=(self.hop_length, 1),
        )
        left = (self.n_fft - self.win_length) // 2
        right = self.n_fft - self.win_length - left
        window = mx.pad(self.window, ((left, right),))
        spectrum = mx.fft.rfft(frames * window, axis=-1)
        power = mx.square(mx.abs(spectrum)).astype(mx.float32)
        mel = self.fb[0] @ mx.transpose(power, (1, 0))
        features = mx.transpose(mx.log(mel + self.log_zero_guard_value), (1, 0))

        # NeMo reports floor(samples / hop) valid frames and masks the final
        # center-padded STFT frame when present.
        valid_frames = wave.shape[0] // self.hop_length
        valid = mx.arange(frame_count) < valid_frames
        features = mx.where(valid[:, None], features, self.pad_value)
        return features[None, :, :].astype(mx.float32), mx.array(
            [valid_frames], dtype=mx.int32
        )


class NemotronPreprocessor(nn.Module):
    """Checkpoint-compatible wrapper retaining ``preprocessor.featurizer``."""

    def __init__(self, args: PreprocessArgs | None = None) -> None:
        super().__init__()
        self.args = args or PreprocessArgs()
        self.featurizer = NemotronFeatureExtractor(
            sample_rate=self.args.sample_rate,
            n_fft=self.args.n_fft,
            win_length=self.args.win_length,
            hop_length=self.args.hop_length,
            n_mels=self.args.features,
            preemphasis=self.args.preemph,
            dither=self.args.dither,
            normalize=self.args.normalize,
            log_zero_guard_value=self.args.log_zero_guard_value,
            pad_value=self.args.pad_value,
        )

    def __call__(self, waveform: mx.array | np.ndarray) -> tuple[mx.array, mx.array]:
        return self.featurizer(waveform)


__all__ = ["NemotronFeatureExtractor", "NemotronPreprocessor"]
