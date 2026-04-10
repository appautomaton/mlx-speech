"""Unified TTS output type and model protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx


@dataclass(frozen=True)
class TTSOutput:
    """Unified output from any TTS model."""

    waveform: mx.array
    sample_rate: int


class TTSModel(Protocol):
    """Protocol for all TTS model wrappers."""

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> TTSOutput: ...
