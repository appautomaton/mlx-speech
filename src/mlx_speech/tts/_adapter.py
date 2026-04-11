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
    """Protocol for all TTS model wrappers.

    Not every backend uses every kwarg. Adapters silently ignore kwargs that
    don't apply. Adapters that require a specific kwarg (e.g. Step Audio edit
    mode requires ``edit_type``) raise ``ValueError`` when it is missing.
    """

    def generate(
        self,
        text: str | None = None,
        *,
        # Voice cloning
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        # Generation budget
        max_new_tokens: int | None = None,
        # Audio editing (Step Audio)
        edit_type: str | None = None,
        edit_info: str | None = None,
        # Sound effect
        duration_seconds: float | None = None,
        # Backend escape hatch
        **kwargs,
    ) -> TTSOutput: ...
