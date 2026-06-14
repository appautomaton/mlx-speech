"""Fish S2 Pro TTS adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.fish_s2_pro import FishS2ProRuntime, PreparedReference


class FishS2ProAdapter:
    def __init__(self, runtime: FishS2ProRuntime):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> FishS2ProAdapter:
        return cls(FishS2ProRuntime.from_dir(model_dir))

    def prepare_reference(
        self,
        reference_audio: str | Path,
        *,
        reference_text: str,
    ) -> PreparedReference:
        """Preprocess a voice reference once for reuse across ``generate`` calls.

        Runs audio loading and codec encoding a single time and returns a handle.
        Pass it back as ``reference_audio`` to clone the same voice across many
        calls without re-encoding the reference each time::

            voice = model.prepare_reference("ref.wav", reference_text="...")
            for line in lines:
                model.generate(line, reference_audio=voice)
        """
        if isinstance(reference_audio, mx.array):
            raise TypeError(
                "Fish S2 Pro expects reference_audio as a file path, not mx.array."
            )
        return self._runtime.encode_reference(str(reference_audio), reference_text)

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | PreparedReference | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> TTSOutput:
        ref_audio = reference_audio
        if ref_audio is not None and not isinstance(ref_audio, PreparedReference):
            if isinstance(ref_audio, mx.array):
                raise TypeError(
                    "Fish S2 Pro expects reference_audio as a file path, not mx.array."
                )
            ref_audio = str(ref_audio)

        result = self._runtime.synthesize(
            text,
            reference_audio=ref_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens or 1024,
        )
        return TTSOutput(waveform=result.waveform, sample_rate=result.sample_rate)
