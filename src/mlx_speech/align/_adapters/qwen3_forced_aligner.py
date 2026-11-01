"""Qwen3-ForcedAligner adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ...generation.qwen3_forced_aligner import Qwen3ForcedAligner
from .._adapter import AlignItem


class Qwen3ForcedAlignerAdapter:
    def __init__(self, runtime: Qwen3ForcedAligner):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Qwen3ForcedAlignerAdapter":
        return cls(Qwen3ForcedAligner.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        text: str,
        *,
        language: str,
        sample_rate: int = 16000,
    ) -> list[AlignItem]:
        result = self._runtime.align(audio, text, language=language, sample_rate=sample_rate)
        return [AlignItem(*span) for span in result.spans]


__all__ = ["Qwen3ForcedAlignerAdapter"]
