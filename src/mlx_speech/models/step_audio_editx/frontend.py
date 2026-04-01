"""CosyVoice frontend helpers for Step-Audio-EditX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...audio import resample_audio
from ..step_audio_tokenizer.config import _parse_scalar
from ..step_audio_tokenizer.processor import _build_slaney_mel_filters, _periodic_hann_window

if TYPE_CHECKING:
    from .campplus import LoadedStepAudioCampPlusModel


@dataclass(frozen=True)
class StepAudioCosyVoiceMelConfig:
    num_mels: int = 80
    n_fft: int = 1920
    hop_size: int = 480
    win_size: int = 1920
    sampling_rate: int = 24000
    fmin: float = 0.0
    fmax: float = 8000.0

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "StepAudioCosyVoiceMelConfig":
        mel_conf = _extract_yaml_section(path, "mel_conf")
        return cls(
            num_mels=int(mel_conf.get("num_mels", 80)),
            n_fft=int(mel_conf.get("n_fft", 1920)),
            hop_size=int(mel_conf.get("hop_size", 480)),
            win_size=int(mel_conf.get("win_size", 1920)),
            sampling_rate=int(mel_conf.get("sampling_rate", 24000)),
            fmin=float(mel_conf.get("fmin", 0.0)),
            fmax=float(mel_conf.get("fmax", 8000.0)),
        )


def _extract_yaml_section(path: str | Path, section_name: str) -> dict[str, object]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = f"{section_name}:"
    start_index: int | None = None
    for index, raw_line in enumerate(lines):
        if raw_line.strip() == header:
            start_index = index + 1
            break
    if start_index is None:
        raise ValueError(f"Missing {section_name} in CosyVoice config: {path}")

    result: dict[str, object] = {}
    for raw_line in lines[start_index:]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            break
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {section_name}: {stripped!r}")
        key, value = stripped.split(":", 1)
        result[key.strip()] = _parse_scalar(value.strip())
    return result


def resolve_step_audio_cosyvoice_dir(model_dir: str | Path) -> Path:
    resolved = Path(model_dir)
    if resolved.name == "CosyVoice-300M-25Hz":
        return resolved
    candidate = resolved / "CosyVoice-300M-25Hz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"CosyVoice asset directory not found under {resolved}")


def _stft_magnitude_padded(
    waveform: np.ndarray,
    *,
    n_fft: int,
    hop_size: int,
    win_size: int,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform with shape (samples,), got {waveform.shape}.")
    if waveform.shape[0] < 2:
        raise ValueError("CosyVoice frontend requires at least 2 samples.")

    pad = int((n_fft - hop_size) / 2)
    padded = np.pad(waveform.astype(np.float32, copy=False), (pad, pad), mode="reflect")
    window = _periodic_hann_window(win_size)
    frame_count = 1 + (padded.shape[0] - n_fft) // hop_size
    n_freqs = n_fft // 2 + 1
    magnitude = np.zeros((n_freqs, frame_count), dtype=np.float32)

    for frame_idx in range(frame_count):
        start = frame_idx * hop_size
        frame = padded[start : start + n_fft].copy()
        frame[:win_size] *= window
        if win_size < n_fft:
            frame[win_size:] = 0.0
        spectrum = np.fft.rfft(frame, n=n_fft)
        magnitude[:, frame_idx] = np.sqrt((np.abs(spectrum) ** 2).astype(np.float32) + 1e-9)

    return magnitude


def mel_spectrogram(
    waveform: np.ndarray,
    *,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    magnitude = _stft_magnitude_padded(
        waveform,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
    )
    mel_filters = _build_slaney_mel_filters(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = mel_filters @ magnitude
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)


class StepAudioCosyVoiceFrontEnd:
    """MLX/NumPy frontend for the non-stream CosyVoice prompt path."""

    def __init__(
        self,
        config: StepAudioCosyVoiceMelConfig,
        *,
        cosyvoice_dir: Path | None = None,
        campplus_model: "LoadedStepAudioCampPlusModel | None" = None,
    ):
        self.config = config
        self.cosyvoice_dir = cosyvoice_dir
        self._campplus_model = campplus_model

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "StepAudioCosyVoiceFrontEnd":
        cosyvoice_dir = resolve_step_audio_cosyvoice_dir(model_dir)
        config = StepAudioCosyVoiceMelConfig.from_yaml_path(cosyvoice_dir / "cosyvoice.yaml")
        return cls(config, cosyvoice_dir=cosyvoice_dir)

    def extract_speech_feat(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = waveform[0]
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                waveform = waveform.mean(axis=0, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform, got {waveform.shape}.")

        if sample_rate != self.config.sampling_rate:
            waveform = np.asarray(
                resample_audio(
                    waveform,
                    orig_sample_rate=sample_rate,
                    target_sample_rate=self.config.sampling_rate,
                ),
                dtype=np.float32,
            )

        speech_feat = mel_spectrogram(
            waveform,
            n_fft=self.config.n_fft,
            num_mels=self.config.num_mels,
            sampling_rate=self.config.sampling_rate,
            hop_size=self.config.hop_size,
            win_size=self.config.win_size,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        ).T[None, :, :]
        speech_feat_len = np.asarray([speech_feat.shape[1]], dtype=np.int64)
        return speech_feat.astype(np.float32), speech_feat_len

    def _ensure_campplus_model(self) -> "LoadedStepAudioCampPlusModel":
        if self._campplus_model is None:
            if self.cosyvoice_dir is None:
                raise ValueError("CosyVoice frontend was created without a model directory.")
            from .campplus import load_step_audio_campplus_model

            self._campplus_model = load_step_audio_campplus_model(self.cosyvoice_dir)
        return self._campplus_model

    def extract_spk_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        campplus_model = self._ensure_campplus_model()
        return campplus_model.runtime.extract_embedding(audio, sample_rate)


__all__ = [
    "StepAudioCosyVoiceFrontEnd",
    "StepAudioCosyVoiceMelConfig",
    "mel_spectrogram",
    "resolve_step_audio_cosyvoice_dir",
]
