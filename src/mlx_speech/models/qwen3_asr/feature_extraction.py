"""Whisper-style log-mel feature extraction for Qwen3-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Qwen3ASRFeatureBatch:
    """Feature extractor output consumed by the Qwen3-ASR processor/model."""

    input_features: np.ndarray
    feature_attention_mask: np.ndarray

    @property
    def audio_lengths(self) -> np.ndarray:
        return _get_feat_extract_output_lengths(
            self.feature_attention_mask.sum(axis=-1)
        )


@dataclass(frozen=True)
class Qwen3ASRFeatureExtractor:
    """Pure NumPy frontend matching the Qwen3-ASR WhisperFeatureExtractor setup."""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 128
    chunk_length: int = 30
    padding_value: float = 0.0
    dither: float = 0.0

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Qwen3ASRFeatureExtractor":
        config_path = Path(model_dir) / "preprocessor_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Qwen3-ASR preprocessor config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            sample_rate=int(payload.get("sampling_rate", 16000)),
            n_fft=int(payload.get("n_fft", 400)),
            hop_length=int(payload.get("hop_length", 160)),
            n_mels=int(payload.get("feature_size", 128)),
            chunk_length=int(payload.get("chunk_length", 30)),
            padding_value=float(payload.get("padding_value", 0.0)),
            dither=float(payload.get("dither", 0.0)),
        )

    @property
    def n_samples(self) -> int:
        return self.chunk_length * self.sample_rate

    @property
    def nb_max_frames(self) -> int:
        return self.n_samples // self.hop_length

    def __call__(
        self,
        audio: np.ndarray | Any | str | Path | list[np.ndarray | Any | str | Path],
        *,
        sample_rate: int = 16000,
    ) -> Qwen3ASRFeatureBatch:
        waveforms = self._normalize_inputs(audio, sample_rate=sample_rate)
        if not waveforms:
            raise ValueError("Qwen3-ASR requires at least one audio input.")
        lengths = np.array([wav.shape[0] for wav in waveforms], dtype=np.int64)
        if np.any(lengths <= 0):
            raise ValueError("Qwen3-ASR requires non-empty audio waveforms.")

        max_length = int(lengths.max())
        padded = np.full(
            (len(waveforms), max_length),
            self.padding_value,
            dtype=np.float32,
        )
        for idx, waveform in enumerate(waveforms):
            padded[idx, : waveform.shape[0]] = waveform

        features = np.stack(
            [self._extract_log_mel(waveform) for waveform in padded],
            axis=0,
        )
        feature_frames = features.shape[-1]
        feature_attention_mask = (
            np.arange(feature_frames, dtype=np.int64)[None, :]
            < (lengths // self.hop_length)[:, None]
        ).astype(np.int64)
        return Qwen3ASRFeatureBatch(
            input_features=features.astype(np.float32, copy=False),
            feature_attention_mask=feature_attention_mask,
        )

    def preflight_shape(self, num_samples: int) -> tuple[int, int]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        feature_frames = num_samples // self.hop_length
        audio_tokens = int(_get_feat_extract_output_lengths(feature_frames))
        return feature_frames, audio_tokens

    def _normalize_inputs(
        self,
        audio: np.ndarray | Any | str | Path | list[np.ndarray | Any | str | Path],
        *,
        sample_rate: int,
    ) -> list[np.ndarray]:
        if isinstance(audio, list):
            return [self._audio_to_numpy(item, sample_rate=sample_rate) for item in audio]
        return [self._audio_to_numpy(audio, sample_rate=sample_rate)]

    def _audio_to_numpy(
        self,
        audio: np.ndarray | Any | str | Path,
        *,
        sample_rate: int,
    ) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            from ...audio import load_audio

            waveform, loaded_sample_rate = load_audio(
                audio,
                sample_rate=self.sample_rate,
                mono=True,
            )
            if loaded_sample_rate != self.sample_rate:
                raise ValueError(
                    f"Qwen3-ASR requires {self.sample_rate} Hz audio; got {loaded_sample_rate} Hz."
                )
            return np.array(waveform, dtype=np.float32)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Qwen3-ASR requires {self.sample_rate} Hz audio; got {sample_rate} Hz. "
                "Resample before calling the feature extractor."
            )

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(
                f"Qwen3-ASR expects a 1D mono waveform, got shape {waveform.shape}."
            )
        return waveform

    def _extract_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        mel_filters = _get_mel_filters(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )
        log_spec = _spectrogram(
            waveform,
            window=_hann_window(self.n_fft),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            mel_filters=mel_filters,
            dither=self.dither,
        )
        log_spec = log_spec[:, :-1]
        if log_spec.size == 0:
            return log_spec.astype(np.float32)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.astype(np.float32)


def _get_feat_extract_output_lengths(input_lengths: int | np.ndarray) -> int | np.ndarray:
    """Reference Qwen3-ASR output length formula after the Conv2D audio tower."""

    lengths = np.asarray(input_lengths, dtype=np.int64)
    input_lengths_leave = lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (lengths // 100) * 13
    )
    output_lengths = np.maximum(output_lengths, 0)
    if np.isscalar(input_lengths):
        return int(output_lengths)
    return output_lengths


_MEL_FILTER_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _get_mel_filters(*, sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (sample_rate, n_fft, n_mels)
    if key not in _MEL_FILTER_CACHE:
        _MEL_FILTER_CACHE[key] = _mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=n_mels,
            sampling_rate=sample_rate,
        )
    return _MEL_FILTER_CACHE[key]


def _hertz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def _mel_to_hertz(mels: float | np.ndarray) -> float | np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def _mel_filter_bank(
    *,
    num_frequency_bins: int,
    num_mel_filters: int,
    sampling_rate: int,
) -> np.ndarray:
    mel_min = _hertz_to_mel(0.0)
    mel_max = _hertz_to_mel(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters.astype(np.float32)


def _create_triangular_filter_bank(
    fft_freqs: np.ndarray,
    filter_freqs: np.ndarray,
) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes))


def _hann_window(length: int) -> np.ndarray:
    return np.hanning(length + 1)[:-1].astype(np.float32)


def _spectrogram(
    waveform: np.ndarray,
    *,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    mel_filters: np.ndarray,
    dither: float,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must be 1D, got {waveform.shape}.")
    waveform = np.pad(
        waveform.astype(np.float64),
        [(frame_length // 2, frame_length // 2)],
        mode="reflect",
    )
    window = window.astype(np.float64)
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))
    num_frequency_bins = frame_length // 2 + 1
    spectrum = np.empty((num_frames, num_frequency_bins), dtype=np.float32)
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        frame = waveform[start : start + frame_length].copy()
        if dither != 0.0:
            frame += dither * np.random.randn(frame_length)
        frame *= window
        fft = np.fft.rfft(frame)
        spectrum[frame_idx] = (np.abs(fft, dtype=np.float64) ** 2).astype(np.float32)

    mel_spec = np.maximum(1e-10, mel_filters.T @ spectrum.T)
    return np.log10(mel_spec).astype(np.float32)
