"""Waveform → log-mel spectrogram (`AudioProcessor`).

This is the front-end used to encode voice-reference audio for IC-LoRA
conditioning. The main TTS path does NOT need this — only the
voice-reference encoding does. We implement it here in pure MLX/NumPy
because we cannot import torchaudio at runtime.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/ops.py:8-55`

Parameters (resolved from `audio-components` metadata):

    target_sample_rate = 16_000
    n_fft = win_length = 1024
    hop_length = 160
    n_mels = 64
    f_min = 0
    f_max = target_sample_rate / 2 = 8_000
    window_fn = hann_window
    center = True       # NOT causal in the actual ops.py
    pad_mode = "reflect"
    power = 1.0         # magnitude (not power)
    mel_scale = "slaney"
    norm = "slaney"
    log: log(clamp(spec, min=1e-5))
    output permute: (B, C, T_mel, n_mels)

The plan documents the `audio-components` config as ``causal_padding=3,
causal=True``, but the actual ``ops.AudioProcessor`` uses centered Hann +
reflect padding — the causal settings in config are training-time leftovers
that the inference code path does not touch.

The main TTS path (decoder + vocoder) does NOT touch this.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


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
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Return torchaudio-compatible Slaney mel filters as ``[n_mels, n_freqs]``."""
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    m_pts = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    f_pts = np.array([_mel_to_hz(mel) for mel in m_pts], dtype=np.float64)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    slopes = f_pts[:, None] - all_freqs[None, :]
    for i in range(n_mels):
        down = -slopes[i] / (f_pts[i + 1] - f_pts[i])
        up = slopes[i + 2] / (f_pts[i + 2] - f_pts[i + 1])
        filters[i] = np.maximum(0.0, np.minimum(down, up))

    # torchaudio norm="slaney": area normalize in Hz.
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    filters *= enorm[:, None]
    return filters.astype(np.float32)


def _hann_window_periodic(win_length: int, n_fft: int) -> np.ndarray:
    """Match ``torch.hann_window(win_length, periodic=True)`` padded for STFT."""
    n = np.arange(win_length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / float(win_length))
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        window = np.pad(window, (left, right), mode="constant")
    return window.astype(np.float32)


def _resample_linear(waveform: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sample_rate == target_sample_rate or waveform.shape[-1] == 0:
        return waveform.astype(np.float32, copy=False)

    duration = waveform.shape[-1] / float(orig_sample_rate)
    target_samples = max(1, int(round(duration * target_sample_rate)))
    source_positions = np.linspace(0.0, duration, num=waveform.shape[-1], endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration, num=target_samples, endpoint=False, dtype=np.float64)

    flat = waveform.reshape(-1, waveform.shape[-1])
    out = np.empty((flat.shape[0], target_samples), dtype=np.float32)
    for row, samples in enumerate(flat):
        out[row] = np.interp(target_positions, source_positions, samples).astype(np.float32)
    return out.reshape(*waveform.shape[:-1], target_samples)


class AudioProcessor:
    """Mel front-end for voice-reference encoding."""

    def __init__(
        self,
        *,
        target_sample_rate: int = 16_000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else target_sample_rate / 2.0

        self._window = _hann_window_periodic(win_length, n_fft)
        self._mel_filters = _slaney_mel_filters(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
        )

    def waveform_to_mel(self, waveform: mx.array, sample_rate: int) -> mx.array:
        """Convert ``[B, C, samples]`` waveform to log-mel ``[B, C, T, 64]``."""
        if waveform.ndim != 3:
            raise ValueError(f"waveform_to_mel expects [B, C, samples]; got {waveform.shape}")

        wav = np.asarray(waveform.astype(mx.float32), dtype=np.float32)
        wav = _resample_linear(wav, int(sample_rate), self.target_sample_rate)
        if wav.shape[-1] == 0:
            raise ValueError("waveform_to_mel requires at least one audio sample")

        pad = self.n_fft // 2
        padded = np.pad(wav, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        frame_count = 1 + (padded.shape[-1] - self.n_fft) // self.hop_length
        if frame_count <= 0:
            raise ValueError(
                f"waveform is too short for n_fft={self.n_fft}: {wav.shape[-1]} samples"
            )

        shape = (*padded.shape[:-1], frame_count, self.n_fft)
        strides = (
            *padded.strides[:-1],
            padded.strides[-1] * self.hop_length,
            padded.strides[-1],
        )
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)
        windowed = frames * self._window.reshape((1, 1, 1, self.n_fft))
        spectrum = np.fft.rfft(windowed, n=self.n_fft, axis=-1)
        magnitude = np.abs(spectrum).astype(np.float32)
        mel = np.einsum("bctf,mf->bctm", magnitude, self._mel_filters, optimize=True)
        log_mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)
        return mx.array(log_mel, dtype=mx.float32)


__all__ = ["AudioProcessor"]
