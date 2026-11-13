"""Qwen3-ASR adapter."""

from __future__ import annotations

from collections.abc import Sequence
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

    def generate_batch(
        self,
        audios: Sequence[np.ndarray | mx.array | str | Path],
        *,
        sample_rate: int = 16000,
        language: str | None | Sequence[str | None] = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
    ) -> list[ASROutput]:
        """Transcribe a list/tuple in input order; see ``transcribe_batch`` for input rules."""
        results = self._runtime.transcribe_batch(
            audios,
            sample_rate=sample_rate,
            language=language,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        return [ASROutput(text=result.text, language=result.language) for result in results]
