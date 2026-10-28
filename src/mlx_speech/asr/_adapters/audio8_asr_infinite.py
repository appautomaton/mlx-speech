"""Unified ASR adapter for Audio8-ASR-Infinite."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ...models.audio8_asr_infinite.model import Audio8ASRModel
from .._adapter import ASROutput


class Audio8ASRAdapter:
    def __init__(self, runtime: Audio8ASRModel) -> None:
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Audio8ASRAdapter":
        return cls(Audio8ASRModel.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        **kwargs,
    ) -> ASROutput:
        if language is None:
            raise ValueError("Audio8 requires language='zh' or 'en'")
        if not isinstance(audio, (str, Path)) and sample_rate != 16000:
            raise ValueError(f"Audio8 requires 16000 Hz arrays; got {sample_rate}")
        if isinstance(audio, (str, Path)):
            from ...audio import load_audio

            waveform, sample_rate = load_audio(audio, sample_rate=16000, mono=True)
        elif isinstance(audio, mx.array):
            waveform = audio.astype(mx.float32)
        else:
            waveform = mx.array(np.asarray(audio, dtype=np.float32))
        result = self._runtime.transcribe(
            waveform, language=language, **kwargs
        )
        return ASROutput(text=result.text, language=result.language)

    def stream_session(
        self,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        att_context_size: tuple[int, int] | list[int] | None = None,
        **kwargs,
    ):
        if sample_rate != 16000:
            raise ValueError(f"Audio8 streaming requires 16000 Hz; got {sample_rate}")
        if language is None:
            raise ValueError("Audio8 requires language='zh' or 'en'")
        if att_context_size is not None:
            raise ValueError("Audio8 uses target_delay_ms instead of att_context_size")
        return self._runtime.stream_session(language=language, **kwargs)


__all__ = ["Audio8ASRAdapter"]
