"""Processor helpers for the Step-Audio tokenizer runtime slice."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from ...audio import resample_audio
from .checkpoint import StepAudioTokenizerAssets, load_step_audio_tokenizer_assets

_LOG_FLOOR = 1e-10

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


def _build_slaney_mel_filters(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    nyquist = sample_rate / 2.0
    mel_fmin = float(fmin)
    mel_fmax = nyquist if fmax is None else float(fmax)
    if mel_fmin < 0.0:
        raise ValueError(f"fmin must be non-negative, got {mel_fmin}.")
    if mel_fmax <= mel_fmin:
        raise ValueError(f"fmax must be greater than fmin, got {mel_fmax} <= {mel_fmin}.")
    if mel_fmax > nyquist:
        raise ValueError(f"fmax must not exceed Nyquist ({nyquist}), got {mel_fmax}.")

    fftfreqs = np.linspace(0.0, nyquist, n_freqs, dtype=np.float64)
    mel_min = _hz_to_mel(mel_fmin)
    mel_max = _hz_to_mel(mel_fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.array([_mel_to_hz(value) for value in mel_points], dtype=np.float64)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    ramps = hz_points[:, None] - fftfreqs[None, :]
    for idx in range(n_mels):
        lower = -ramps[idx] / (hz_points[idx + 1] - hz_points[idx])
        upper = ramps[idx + 2] / (hz_points[idx + 2] - hz_points[idx + 1])
        filters[idx] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filters *= enorm[:, None]
    return filters.astype(np.float32)


def _periodic_hann_window(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"Window length must be positive, got {length}.")
    positions = np.arange(length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos((2.0 * np.pi * positions) / float(length))).astype(np.float32)


def _stft_power_reflect_padded(
    waveform: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape (samples,), got {waveform.shape}.")
    if waveform.shape[0] < 2:
        raise ValueError("Step-Audio semantic chunks require at least 2 samples.")

    pad = n_fft // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")
    window = _periodic_hann_window(n_fft)
    n_frames = 1 + (padded.shape[0] - n_fft) // hop_length
    n_freqs = n_fft // 2 + 1

    power = np.zeros((n_freqs, n_frames), dtype=np.float32)
    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        frame = padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power[:, frame_idx] = (np.abs(spectrum) ** 2).astype(np.float32)

    if power.shape[1] > 0:
        power = power[:, :-1]
    return power


def _signal_to_frame_nonsilent(
    waveform: np.ndarray,
    *,
    top_db: float,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if waveform.size == 0:
        return np.zeros((0,), dtype=bool)

    pad = frame_length // 2
    padded = np.pad(waveform, (pad, pad), mode="constant")
    frame_count = 1 + (padded.shape[0] - frame_length) // hop_length
    if frame_count <= 0:
        return np.zeros((0,), dtype=bool)

    rms = np.zeros((frame_count,), dtype=np.float32)
    for frame_idx in range(frame_count):
        start = frame_idx * hop_length
        frame = padded[start : start + frame_length]
        rms[frame_idx] = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    ref_value = float(np.max(rms))
    if ref_value <= 0.0:
        return np.zeros((frame_count,), dtype=bool)

    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    db -= 20.0 * math.log10(max(ref_value, 1e-10))
    return db > (-float(top_db))


def _trim_silence(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    top_db: float,
    frame_length: int,
    hop_length: int,
    keep_left_seconds: float,
    keep_right_seconds: float,
    output_hop_samples: int,
) -> np.ndarray:
    non_silent = _signal_to_frame_nonsilent(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    nonzero = np.flatnonzero(non_silent)
    if nonzero.size > 0:
        start = int(nonzero[0] * hop_length)
        end = min(waveform.shape[0], int((nonzero[-1] + 1) * hop_length))
    else:
        start, end = 0, 0

    num_frames = int(math.ceil(max(0, end - start) / float(output_hop_samples)))
    left_keep = int(keep_left_seconds * sample_rate)

    start_idx = start - left_keep
    trimmed = waveform
    if start_idx > 0:
        trimmed = trimmed[start_idx:]
    else:
        trimmed = np.pad(trimmed, (abs(start_idx), 0), mode="constant", constant_values=0.0)

    out_len = int(
        num_frames * output_hop_samples
        + (keep_left_seconds + keep_right_seconds) * sample_rate
    )
    if out_len < trimmed.shape[0]:
        trimmed = trimmed[:out_len]
    else:
        trimmed = np.pad(
            trimmed,
            (0, out_len - trimmed.shape[0]),
            mode="constant",
            constant_values=0.0,
        )
    return trimmed.astype(np.float32, copy=False)


def _to_mono_float32(audio: np.ndarray | mx.array) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        if waveform.shape[0] == 1:
            return waveform[0].astype(np.float32, copy=False)
        if waveform.shape[1] == 1:
            return waveform[:, 0].astype(np.float32, copy=False)
        if waveform.shape[0] <= waveform.shape[1]:
            return waveform.mean(axis=0, dtype=np.float32)
        return waveform.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Expected mono or simple multi-channel audio, got {waveform.shape}.")


def _energy_normalize(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    if peak <= 0.0:
        return waveform.astype(np.float32, copy=False)
    return (waveform / max(peak, 0.01) * 0.999).astype(np.float32, copy=False)


@dataclass(frozen=True)
class StepAudioVQ06Chunk:
    waveform: np.ndarray
    features: np.ndarray
    feature_length: int
    duration_seconds: float
    expected_token_length: int


class StepAudioTokenizerProcessor:
    """Pure-MLX/NumPy tokenizer preprocessing utilities for Step-Audio."""

    def __init__(self, assets: StepAudioTokenizerAssets):
        self.assets = assets
        self.config = assets.config
        self._vq06_mel_filters = _build_slaney_mel_filters(
            sample_rate=self.config.vq06_sample_rate,
            n_fft=self.config.vq06_n_fft,
            n_mels=self.config.vq06_num_mels,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path | None = None) -> "StepAudioTokenizerProcessor":
        return cls(load_step_audio_tokenizer_assets(model_dir))

    def preprocess_wav(
        self,
        audio: np.ndarray | mx.array,
        sample_rate: int,
        *,
        enable_trim: bool = True,
        energy_norm: bool = True,
    ) -> np.ndarray:
        waveform = _to_mono_float32(audio)
        if int(sample_rate) != self.config.vq02_sample_rate:
            waveform = np.asarray(
                resample_audio(
                    mx.array(waveform, dtype=mx.float32),
                    orig_sample_rate=int(sample_rate),
                    target_sample_rate=self.config.vq02_sample_rate,
                ),
                dtype=np.float32,
            )
        if energy_norm:
            waveform = _energy_normalize(waveform)
        if enable_trim:
            waveform = _trim_silence(
                waveform,
                sample_rate=self.config.vq02_sample_rate,
                top_db=self.config.trim_top_db,
                frame_length=self.config.trim_frame_length,
                hop_length=self.config.trim_hop_length,
                keep_left_seconds=self.config.trim_keep_left_seconds,
                keep_right_seconds=self.config.trim_keep_right_seconds,
                output_hop_samples=self.config.trim_output_hop_samples,
            )
        return waveform.astype(np.float32, copy=False)

    def cluster_linguistic_features(self, features: np.ndarray) -> list[int]:
        dense = np.asarray(features, dtype=np.float32)
        if dense.ndim == 3:
            if dense.shape[0] != 1:
                raise ValueError(
                    "Expected batched linguistic features with batch size 1, got "
                    f"{dense.shape}."
                )
            dense = dense[0]
        if dense.ndim != 2:
            raise ValueError(
                "Expected linguistic features with shape (frames, dim) or (1, frames, dim), "
                f"got {dense.shape}."
            )

        codebook = np.asarray(self.assets.linguistic_codebook, dtype=np.float32)
        if dense.shape[1] != codebook.shape[1]:
            raise ValueError(
                "Linguistic feature dim does not match the tokenizer codebook: "
                f"{dense.shape[1]} vs {codebook.shape[1]}."
            )

        sample_norm = np.sum(dense * dense, axis=1, keepdims=True)
        codebook_norm = np.sum(codebook * codebook, axis=1)[None, :]
        distances = sample_norm + codebook_norm - 2.0 * (dense @ codebook.T)
        return distances.argmin(axis=1).astype(np.int32).tolist()

    def dump_label(self, samples: Sequence[np.ndarray]) -> list[list[int]]:
        return [self.cluster_linguistic_features(sample) for sample in samples]

    def split_vq06_audio(self, audio: np.ndarray | mx.array) -> list[np.ndarray]:
        waveform = _to_mono_float32(audio)
        max_samples = int(round(self.config.vq06_max_chunk_seconds * self.config.vq06_sample_rate))
        if waveform.shape[0] <= max_samples:
            return [waveform]

        chunks: list[np.ndarray] = []
        start = 0
        while start < waveform.shape[0]:
            end = min(start + max_samples, waveform.shape[0])
            chunk = waveform[start:end]
            if chunk.shape[0] >= self.config.vq06_min_chunk_samples:
                chunks.append(chunk.astype(np.float32, copy=False))
            start = end
        return chunks

    def compute_vq06_log_mel_spectrogram(self, audio: np.ndarray | mx.array) -> np.ndarray:
        waveform = _to_mono_float32(audio)
        power = _stft_power_reflect_padded(
            waveform,
            n_fft=self.config.vq06_n_fft,
            hop_length=self.config.vq06_hop_length,
        )
        mel_spec = self._vq06_mel_filters @ power
        log_spec = np.log10(np.maximum(mel_spec, _LOG_FLOOR))
        log_spec = np.maximum(log_spec, float(log_spec.max()) - 8.0)
        return ((log_spec + 4.0) / 4.0).astype(np.float32)

    def prepare_vq06_chunks(self, audio: np.ndarray | mx.array) -> list[StepAudioVQ06Chunk]:
        chunks = self.split_vq06_audio(audio)
        prepared: list[StepAudioVQ06Chunk] = []
        for chunk in chunks:
            features = self.compute_vq06_log_mel_spectrogram(chunk)
            duration_seconds = round(chunk.shape[0] / float(self.config.vq06_sample_rate), 2)
            prepared.append(
                StepAudioVQ06Chunk(
                    waveform=chunk,
                    features=features,
                    feature_length=int(features.shape[-1]),
                    duration_seconds=duration_seconds,
                    expected_token_length=int(
                        round(duration_seconds * float(self.config.vq06_token_rate_hz))
                    ),
                )
            )
        return prepared


__all__ = [
    "StepAudioTokenizerProcessor",
    "StepAudioVQ06Chunk",
]
