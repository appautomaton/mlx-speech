"""Fish S2 Pro TTS adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.fish_s2_pro import FishS2ProRuntime


class FishS2ProAdapter:
    def __init__(self, runtime: FishS2ProRuntime):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> FishS2ProAdapter:
        return cls(FishS2ProRuntime.from_dir(model_dir))

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> TTSOutput:
        ref_audio = None
        if reference_audio is not None:
            if isinstance(reference_audio, mx.array):
                raise TypeError(
                    "Fish S2 Pro expects reference_audio as a file path, not mx.array."
                )
            ref_audio = str(reference_audio)

        result = self._runtime.synthesize(
            text,
            reference_audio=ref_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens or 1024,
        )
        return TTSOutput(waveform=result.waveform, sample_rate=result.sample_rate)
