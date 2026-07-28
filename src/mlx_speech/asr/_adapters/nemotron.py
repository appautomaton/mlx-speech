"""Unified ASR adapter for Nemotron 3.5 ASR."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ...models.nemotron_asr.model import NemotronASRModel
from .._adapter import ASROutput


class NemotronASRAdapter:
    def __init__(self, runtime: NemotronASRModel) -> None:
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> "NemotronASRAdapter":
        return cls(NemotronASRModel.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        **kwargs,
    ) -> ASROutput:
        if isinstance(audio, (str, Path)):
            from ...audio import load_audio

            waveform, sample_rate = load_audio(
                audio, sample_rate=16_000, mono=True
            )
        elif isinstance(audio, mx.array):
            waveform = audio.astype(mx.float32)
        else:
            waveform = mx.array(np.asarray(audio, dtype=np.float32))
        result = self._runtime.transcribe(
            waveform,
            sample_rate=sample_rate,
            language=language,
            **kwargs,
        )
        return ASROutput(
            text=result.text,
            language=result.detected_language or result.language,
        )

    def stream_session(
        self,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        att_context_size: tuple[int, int] | list[int] | None = None,
    ):
        if sample_rate != 16_000:
            raise ValueError(f"Nemotron streaming requires 16000 Hz; got {sample_rate}")
        return self._runtime.stream_session(
            language=language,
            att_context_size=att_context_size,
        )


__all__ = ["NemotronASRAdapter"]
