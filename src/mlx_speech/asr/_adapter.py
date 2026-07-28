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
        language: str | None = None,
        **kwargs,
    ) -> ASROutput: ...


class ASRStreamSession(Protocol):
    """Persistent token stream returned by streaming-native ASR families."""

    def feed(self, pcm: np.ndarray | mx.array) -> tuple[int, ...]: ...

    def finalize(self) -> tuple[int, ...]: ...


class StreamingASRModel(ASRModel, Protocol):
    """ASR model that additionally supports live arbitrary-chunk input."""

    def stream_session(
        self,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        att_context_size: tuple[int, int] | list[int] | None = None,
    ) -> ASRStreamSession: ...
