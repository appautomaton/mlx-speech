"""Unified ASR output type and model protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class ASROutput:
    """Unified output from any ASR model."""

    text: str
    language: str


class ASRModel(Protocol):
    """Protocol for all ASR model wrappers."""

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str = "en",
        **kwargs,
    ) -> ASROutput: ...
