"""dots.tts adapter for the unified non-streaming TTS API."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Iterator, Literal

import mlx.core as mx

from ...generation.dots_tts import (
    DEFAULT_MAX_AUDIO_PATCHES,
    DotsTTSGenerator,
)
from .._adapter import TTSOutput


class DotsTTSAdapter:
    def __init__(self, generator: DotsTTSGenerator):
        self._generator = generator

    @classmethod
    def from_dir(cls, model_dir: Path) -> "DotsTTSAdapter":
        return cls(DotsTTSGenerator.from_dir(model_dir))

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        max_audio_patches: int | None = None,
        reference_sample_rate: int | None = None,
        solver_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        eos_threshold: float = 0.8,
        template: Literal["tts", "tts_interleave"] = "tts",
        stream_chunk_patches: int = 4,
        **kwargs,
    ) -> TTSOutput:
        chunks = list(
            self.generate_stream(
                text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                max_new_tokens=max_new_tokens,
                max_audio_patches=max_audio_patches,
                reference_sample_rate=reference_sample_rate,
                solver_steps=solver_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                language=language,
                seed=seed,
                eos_threshold=eos_threshold,
                template=template,
                stream_chunk_patches=stream_chunk_patches,
                **kwargs,
            )
        )
        return TTSOutput(
            waveform=mx.concatenate([chunk.waveform for chunk in chunks]),
            sample_rate=self._generator.sample_rate,
        )

    def generate_stream(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        max_audio_patches: int | None = None,
        reference_sample_rate: int | None = None,
        solver_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        eos_threshold: float = 0.8,
        template: Literal["tts", "tts_interleave"] = "tts",
        stream_chunk_patches: int = 4,
        **kwargs,
    ) -> Iterator[TTSOutput]:
        if (
            max_new_tokens is not None
            and max_audio_patches is not None
            and max_new_tokens != max_audio_patches
        ):
            raise ValueError(
                "max_new_tokens and max_audio_patches must match when both are set"
            )
        if (
            isinstance(stream_chunk_patches, bool)
            or not isinstance(stream_chunk_patches, Integral)
            or stream_chunk_patches <= 0
        ):
            raise ValueError("stream_chunk_patches must be a positive integer")
        patch_budget = (
            max_audio_patches
            if max_audio_patches is not None
            else max_new_tokens
        )
        chunks = self._generator.synthesize_stream(
            text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            reference_sample_rate=reference_sample_rate,
            max_audio_patches=(
                DEFAULT_MAX_AUDIO_PATCHES if patch_budget is None else patch_budget
            ),
            solver_steps=solver_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            language=language,
            seed=seed,
            eos_threshold=eos_threshold,
            template=template,
            stream_chunk_patches=int(stream_chunk_patches),
        )
        for chunk in chunks:
            yield TTSOutput(
                waveform=chunk.waveform,
                sample_rate=self._generator.sample_rate,
            )


__all__ = ["DotsTTSAdapter"]
