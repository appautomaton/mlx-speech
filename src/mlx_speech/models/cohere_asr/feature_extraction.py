"""Log-Mel feature extraction for CohereAsr — pure numpy, no torch, no librosa."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Matches upstream LOG_ZERO_GUARD_VALUE = 2**-24
_LOG_ZERO_GUARD = 2**-24
_EPSILON = 1e-5


# ---------------------------------------------------------------------------
# Mel filterbank (replicates librosa.filters.mel with norm="slaney")
# ---------------------------------------------------------------------------

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


def _mel_filters(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Slaney-normalised mel filterbank matching librosa.filters.mel(norm='slaney').

    Returns shape (n_mels, n_fft // 2 + 1) float32.
    """
    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_freqs, dtype=np.float64)

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points], dtype=np.float64)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    ramps = hz_points[:, None] - fftfreqs[None, :]
    for i in range(n_mels):
        lower = -ramps[i] / (hz_points[i + 1] - hz_points[i])
        upper = ramps[i + 2] / (hz_points[i + 2] - hz_points[i + 1])
        filters[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization in Hz.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filters *= enorm[:, None]

    return filters.astype(np.float32)


# Module-level cache so the filterbank is built once.
_MEL_FILTER_CACHE: dict[tuple, np.ndarray] = {}


def _get_mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _MEL_FILTER_CACHE:
        _MEL_FILTER_CACHE[key] = _mel_filters(sr, n_fft, n_mels, fmin, fmax)
    return _MEL_FILTER_CACHE[key]


# ---------------------------------------------------------------------------
# STFT (replicates torch.stft with center=True, pad_mode="constant")
# ---------------------------------------------------------------------------

def _stft_power(
    waveform: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Center-padded STFT power spectrum.

    Args:
        waveform: (T,) float32 mono waveform.
    Returns:
        (n_fft // 2 + 1, frames) float32 power spectrum.
    """
    # Center pad by n_fft // 2 on each side (matches torch.stft center=True default)
    pad = n_fft // 2
    waveform = np.pad(waveform, pad, mode="constant")

    window = np.hanning(win_length).astype(np.float32)
    # Zero-pad window to n_fft if win_length < n_fft
    if win_length < n_fft:
        pad_w = (n_fft - win_length) // 2
        window = np.pad(window, (pad_w, n_fft - win_length - pad_w))

    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    n_freqs = n_fft // 2 + 1

    power = np.zeros((n_freqs, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power[:, i] = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

    return power


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _apply_preemphasis(waveform: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """High-pass preemphasis filter: y[t] = x[t] - coeff * x[t-1]."""
    if coeff == 0.0:
        return waveform
    result = np.empty_like(waveform)
    result[0] = waveform[0]
    result[1:] = waveform[1:] - coeff * waveform[:-1]
    return result


def _apply_dither(waveform: np.ndarray, amount: float = 1e-5) -> np.ndarray:
    """Deterministic dither seeded by waveform length (matches upstream)."""
    if amount <= 0.0:
        return waveform
    rng = np.random.default_rng(seed=len(waveform))
    noise = rng.standard_normal(len(waveform)).astype(np.float32)
    return waveform + amount * noise


def _log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 128,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Compute log-Mel spectrogram.

    Returns:
        (frames, n_mels) float32.
    """
    mel_fb = _get_mel_filters(sr, n_fft, n_mels, 0.0, fmax)
    power = _stft_power(waveform, n_fft, hop_length, win_length)  # (n_freqs, frames)
    mel = mel_fb @ power  # (n_mels, frames)
    log_mel = np.log(mel + _LOG_ZERO_GUARD)
    return log_mel.T  # (frames, n_mels)


# ---------------------------------------------------------------------------
# Long-form chunking
# ---------------------------------------------------------------------------

def _find_split_energy(
    waveform: np.ndarray,
    start: int,
    end: int,
    window_samples: int = 1600,
) -> int:
    segment = waveform[start:end]
    if len(segment) <= window_samples:
        return (start + end) // 2
    min_energy = float("inf")
    best = start
    upper = len(segment) - window_samples
    for i in range(0, upper, window_samples):
        w = segment[i : i + window_samples]
        energy = float(np.sqrt(np.mean(w * w)))
        if energy < min_energy:
            min_energy = energy
            best = start + i
    return best


def split_audio_chunks(
    waveform: np.ndarray,
    sr: int = 16000,
    max_clip_s: float = 35.0,
    overlap_s: float = 5.0,
    min_energy_window_samples: int = 1600,
) -> list[np.ndarray]:
    """Split long audio at energy-based boundaries.

    Returns list of chunks; single-element list if audio is short enough.
    """
    chunk_size = max(1, int(round(max_clip_s * sr)))
    boundary_ctx = max(1, int(round(overlap_s * sr)))
    total = len(waveform)

    if total <= chunk_size:
        return [waveform]

    chunks: list[np.ndarray] = []
    idx = 0
    while idx < total:
        if idx + chunk_size >= total:
            chunks.append(waveform[idx:])
            break
        search_start = max(idx, idx + chunk_size - boundary_ctx)
        search_end = min(idx + chunk_size, total)
        if search_end <= search_start:
            split = idx + chunk_size
        else:
            split = _find_split_energy(
                waveform,
                search_start,
                search_end,
                window_samples=min_energy_window_samples,
            )
        split = max(idx + 1, min(split, total))
        chunks.append(waveform[idx:split])
        idx = split

    return chunks


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

class CohereAsrFeatureExtractor:
    """Extract log-Mel features from raw audio for CohereAsr.

    All computation is numpy-only, no torch or librosa required at inference.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 128,
        fmax: float = 8000.0,
        preemphasis: float = 0.97,
        dither: float = 1e-5,
        max_audio_clip_s: float = 35.0,
        overlap_chunk_s: float = 5.0,
        min_energy_window_samples: int = 1600,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.preemphasis = preemphasis
        self.dither = dither
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_s = overlap_chunk_s
        self.min_energy_window_samples = min_energy_window_samples
        # Pre-build mel filterbank
        _ = _get_mel_filters(sr, n_fft, n_mels, 0.0, fmax)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "CohereAsrFeatureExtractor":
        """Build the feature extractor from checkpoint-side preprocessor settings."""
        model_dir = Path(model_dir)
        preprocessor_path = model_dir / "preprocessor_config.json"
        config_path = model_dir / "config.json"

        preprocessor_payload: dict[str, object] = {}
        if preprocessor_path.exists():
            with preprocessor_path.open(encoding="utf-8") as f:
                preprocessor_payload = json.load(f)

        config_payload: dict[str, object] = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                config_payload = json.load(f)

        if not preprocessor_payload and not config_payload:
            return cls()

        return cls(
            sr=int(preprocessor_payload.get("sampling_rate", config_payload.get("sample_rate", 16000))),
            n_fft=int(preprocessor_payload.get("n_fft", 512)),
            hop_length=int(preprocessor_payload.get("n_window_stride", preprocessor_payload.get("hop_length", 160))),
            win_length=int(preprocessor_payload.get("n_window_size", preprocessor_payload.get("win_length", 400))),
            n_mels=int(preprocessor_payload.get("feature_size", preprocessor_payload.get("features", 128))),
            preemphasis=float(preprocessor_payload.get("preemphasis", 0.97)),
            dither=float(preprocessor_payload.get("dither", 1e-5)),
            fmax=float(preprocessor_payload.get("highfreq", 8000.0)),
            max_audio_clip_s=float(config_payload.get("max_audio_clip_s", 35.0)),
            overlap_chunk_s=float(
                config_payload.get("overlap_chunk_s", config_payload.get("overlap_chunk_second", 5.0))
            ),
            min_energy_window_samples=int(config_payload.get("min_energy_window_samples", 1600)),
        )

    def _features_length(self, audio_length: int) -> int:
        """Number of mel frames for a given number of audio samples."""
        return audio_length // self.hop_length

    def __call__(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from a single waveform.

        Args:
            waveform: (T,) float32 mono audio at self.sr.

        Returns:
            features: (frames, n_mels) float32 normalized log-Mel features.
            attention_mask: (frames,) bool — True for valid (non-padded) frames.
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

        # Dither before preemphasis
        waveform = _apply_dither(waveform, self.dither)
        # Preemphasis
        waveform = _apply_preemphasis(waveform, self.preemphasis)

        features = _log_mel_spectrogram(
            waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )  # (frames, n_mels)

        valid_frames = self._features_length(len(waveform))
        valid_frames = min(valid_frames, len(features))
        attention_mask = np.zeros(len(features), dtype=bool)
        attention_mask[:valid_frames] = True

        # Per-sample zero-mean unit-variance normalisation over valid frames only
        valid_feat = features[:valid_frames]
        mean = valid_feat.mean(axis=0, keepdims=True)
        if valid_frames > 1:
            variance = ((valid_feat - mean) ** 2).sum(axis=0, keepdims=True) / (valid_frames - 1)
            std = np.sqrt(variance).astype(np.float32)
        else:
            std = np.zeros_like(mean, dtype=np.float32)
        features = (features - mean) / (std + _EPSILON)
        # Zero out padded frames
        features[valid_frames:] = 0.0

        return features, attention_mask

    def process_audio(
        self,
        waveform: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Process audio, chunking if longer than max_audio_clip_s.

        Returns a list of (features, attention_mask) per chunk.
        """
        fast_path_threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_s)
        if len(waveform) / self.sr <= fast_path_threshold_s:
            return [self(waveform)]

        chunks = split_audio_chunks(
            waveform,
            sr=self.sr,
            max_clip_s=self.max_audio_clip_s,
            overlap_s=self.overlap_chunk_s,
            min_energy_window_samples=self.min_energy_window_samples,
        )
        return [self(chunk) for chunk in chunks]
