"""Qwen3-ASR adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .._adapter import ASROutput
from ...generation.qwen3_asr import Qwen3ASRTranscriber


class Qwen3ASRAdapter:
    def __init__(self, runtime: Qwen3ASRTranscriber):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Qwen3ASRAdapter":
        return cls(Qwen3ASRTranscriber.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        **kwargs,
    ) -> ASROutput:
        result = self._runtime.transcribe(
            audio,
            sample_rate=sample_rate,
            language=language,
            **kwargs,
        )
        return ASROutput(text=result.text, language=result.language)
