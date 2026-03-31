"""Small audio I/O helpers for local v0 validation."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import mlx.core as mx

try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs
    sf = None


def mix_down_mono(samples: mx.array) -> mx.array:
    """Convert audio shaped like (samples,) or (samples, channels) to mono."""

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim == 1:
        return mx.array(waveform, dtype=mx.float32)
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected waveform with shape (samples,) or (samples, channels), got {waveform.shape}."
        )
    return mx.array(np.mean(waveform, axis=1, dtype=np.float32), dtype=mx.float32)


def resample_audio(
    samples: mx.array,
    *,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> mx.array:
    """Resample mono audio with linear interpolation."""

    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape (samples,), got {waveform.shape}.")
    if waveform.size == 0 or orig_sample_rate == target_sample_rate:
        return mx.array(waveform, dtype=mx.float32)

    duration = waveform.shape[0] / float(orig_sample_rate)
    target_samples = max(1, int(round(duration * target_sample_rate)))
    source_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_samples, endpoint=False, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, waveform).astype(np.float32)
    return mx.array(resampled, dtype=mx.float32)


def loudness_normalize(
    samples: mx.array,
    *,
    target_dbfs: float = -20.0,
    gain_range: tuple[float, float] = (-3.0, 3.0),
) -> mx.array:
    """Apply a small loudness correction in dBFS."""

    waveform = samples.astype(mx.float32)
    if waveform.size == 0:
        return waveform

    power = float(mx.mean(waveform * waveform).item())
    current_dbfs = 10.0 * np.log10(power + 1e-9)
    gain = max(gain_range[0], min(target_dbfs - current_dbfs, gain_range[1]))
    factor = 10.0 ** (gain / 20.0)
    return waveform * factor


def load_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[mx.array, int]:
    """Load local audio from disk."""

    if sf is not None:
        waveform, loaded_sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        samples = mx.array(waveform, dtype=mx.float32)
    else:
        with wave.open(str(path), "rb") as wav_file:
            loaded_sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width != 2:
            raise RuntimeError("WAV fallback supports 16-bit PCM only.")
        waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
            waveform = waveform.reshape(-1, channels)
        samples = mx.array(waveform, dtype=mx.float32)
    if mono:
        samples = mix_down_mono(samples)
    if sample_rate is not None and int(loaded_sample_rate) != int(sample_rate):
        samples = resample_audio(
            samples,
            orig_sample_rate=int(loaded_sample_rate),
            target_sample_rate=int(sample_rate),
        )
        loaded_sample_rate = int(sample_rate)
    return samples, int(loaded_sample_rate)


def trim_leading_silence(
    samples: mx.array,
    *,
    sample_rate: int,
    threshold: float = 0.003,
    frame_ms: float = 20.0,
    keep_ms: float = 80.0,
) -> mx.array:
    """Trim leading low-energy audio using a small RMS window."""

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape (samples,), got {waveform.shape}.")
    if waveform.size == 0:
        return samples

    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    keep_samples = max(0, int(sample_rate * keep_ms / 1000.0))

    start_index = 0
    for idx in range(0, waveform.size, frame_size):
        frame = waveform[idx : idx + frame_size]
        if frame.size == 0:
            break
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms >= threshold:
            start_index = max(0, idx - keep_samples)
            break
    else:
        return samples

    return mx.array(waveform[start_index:], dtype=samples.dtype)


def normalize_peak(
    samples: mx.array,
    *,
    target_peak: float = 0.95,
    max_gain: float = 4.0,
) -> mx.array:
    """Scale waveform so peak amplitude approaches `target_peak`."""

    waveform = samples.astype(mx.float32)
    peak = float(mx.max(mx.abs(waveform)).item()) if waveform.size > 0 else 0.0
    if peak <= 0.0:
        return waveform

    gain = min(max_gain, target_peak / peak)
    if gain <= 1.0:
        return waveform
    return waveform * gain


def write_wav(path: str | Path, samples: mx.array, *, sample_rate: int) -> Path:
    """Write a mono float waveform to 16-bit PCM WAV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape (samples,), got {waveform.shape}.")
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    return output_path
