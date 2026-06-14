"""DramaBox TTS adapter.

Wraps the lower-level :class:`DramaBoxModel` behind the unified
``generate(text, ...)`` interface. DramaBox needs a separate Gemma 3 12B
text-encoder backbone, resolved and passed in as ``gemma_dir`` by the loader.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput

# Generation knobs forwarded verbatim to DramaBoxModel.generate via **kwargs.
_PASSTHROUGH = (
    "duration_s",
    "cfg_scale",
    "stg_scale",
    "rescale_scale",
    "modality_scale",
    "steps",
    "seed",
    "denoise_ref",
)


class DramaBoxAdapter:
    def __init__(self, model):
        self._model = model

    @classmethod
    def from_dir(cls, model_dir: Path, *, gemma_dir: str | Path) -> DramaBoxAdapter:
        from ...generation.dramabox import DramaBoxModel

        return cls(DramaBoxModel.from_dir(model_dir, gemma_dir=gemma_dir))

    def generate(
        self,
        text: str | None = None,
        *,
        reference_audio: str | Path | mx.array | None = None,
        duration_seconds: float | None = None,
        **kwargs,
    ) -> TTSOutput:
        if not text:
            raise ValueError("DramaBox requires a non-empty text prompt.")

        gen_kwargs: dict = {}
        # Map the shared `duration_seconds` kwarg onto DramaBox's `duration_s`.
        if duration_seconds is not None:
            gen_kwargs["duration_s"] = duration_seconds
        for key in _PASSTHROUGH:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]

        if reference_audio is not None:
            if isinstance(reference_audio, mx.array):
                raise TypeError(
                    "DramaBox expects reference_audio as a file path, not mx.array."
                )
            gen_kwargs["voice_ref"] = str(reference_audio)

        result = self._model.generate(text, **gen_kwargs)
        return TTSOutput(waveform=result.waveform, sample_rate=result.sample_rate)
