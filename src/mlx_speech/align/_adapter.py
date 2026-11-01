"""Unified forced-alignment output type and model protocol."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol

import mlx.core as mx
import numpy as np


class AlignItem(NamedTuple):
    """One aligned unit: a CJK character or a word, with times in seconds."""

    text: str
    start_s: float
    end_s: float


class AlignModel(Protocol):
    """Protocol for all forced-alignment model wrappers."""

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        text: str,
        *,
        language: str,
        sample_rate: int = 16000,
    ) -> list[AlignItem]: ...
