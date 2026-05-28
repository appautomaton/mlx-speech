"""Reference-audio preparation for DramaBox voice cloning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ReferenceAudio:
    """Encoder-ready reference audio."""

    waveform: mx.array  # [1, 2, samples]
    sample_rate: int


def _resample_channels(waveform: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)

    duration = waveform.shape[0] / float(orig_sample_rate)
    target_samples = max(1, int(round(duration * target_sample_rate)))
    source_positions = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration, num=target_samples, endpoint=False, dtype=np.float64)

    out = np.empty((target_samples, waveform.shape[1]), dtype=np.float32)
    for channel in range(waveform.shape[1]):
        out[:, channel] = np.interp(target_positions, source_positions, waveform[:, channel]).astype(np.float32)
    return out


def _force_stereo(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim != 2:
        raise ValueError(f"expected audio shaped [samples, channels], got {waveform.shape}")
    if waveform.shape[1] == 2:
        return waveform.astype(np.float32, copy=False)
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32, copy=False)


def _crop_or_loop(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        raise ValueError("ref_duration_s must produce at least one sample")
    if waveform.shape[0] == 0:
        raise ValueError("reference audio is empty")
    if waveform.shape[0] >= target_samples:
        return waveform[:target_samples].astype(np.float32, copy=False)

    repeats = int(np.ceil(target_samples / waveform.shape[0]))
    return np.tile(waveform, (repeats, 1))[:target_samples].astype(np.float32, copy=False)


def _peak_normalize_dbfs(waveform: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 0.0:
        raise ValueError("reference audio is silent")
    target_peak = 10.0 ** (target_dbfs / 20.0)
    return (waveform * (target_peak / peak)).astype(np.float32, copy=False)


def prepare_reference_audio(
    path: str | Path,
    *,
    ref_duration_s: float = 10.0,
    target_sample_rate: int = 16_000,
    target_peak_dbfs: float = -4.0,
) -> ReferenceAudio:
    """Load a local reference file and return ``[1, 2, samples]`` float32 audio."""
    ref_path = Path(path)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    waveform, sample_rate = sf.read(str(ref_path), always_2d=True, dtype="float32")
    waveform = _force_stereo(np.asarray(waveform, dtype=np.float32))
    waveform = _resample_channels(waveform, int(sample_rate), target_sample_rate)
    target_samples = int(round(float(ref_duration_s) * target_sample_rate))
    waveform = _crop_or_loop(waveform, target_samples)
    waveform = _peak_normalize_dbfs(waveform, target_peak_dbfs)

    # soundfile uses [samples, channels]; AudioProcessor expects [B, C, samples].
    waveform_bct = waveform.T[None, :, :]
    return ReferenceAudio(waveform=mx.array(waveform_bct, dtype=mx.float32), sample_rate=target_sample_rate)


__all__ = ["ReferenceAudio", "prepare_reference_audio"]
