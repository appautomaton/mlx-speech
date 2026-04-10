"""Cohere ASR adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .._adapter import ASROutput
from ...generation.cohere_asr import CohereAsrModel


class CohereASRAdapter:
    def __init__(self, runtime: CohereAsrModel):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> CohereASRAdapter:
        return cls(CohereAsrModel.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str = "en",
        **kwargs,
    ) -> ASROutput:
        if isinstance(audio, (str, Path)):
            from ...audio import load_audio

            waveform, sample_rate = load_audio(
                audio, sample_rate=16000, mono=True
            )
            audio_np = np.array(waveform, dtype=np.float32)
        elif isinstance(audio, mx.array):
            audio_np = np.array(audio, dtype=np.float32)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        result = self._runtime.transcribe(
            audio_np, sample_rate=sample_rate, language=language, **kwargs
        )
        return ASROutput(text=result.text, language=result.language)
