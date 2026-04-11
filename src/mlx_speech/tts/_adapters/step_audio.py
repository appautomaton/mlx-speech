"""Step-Audio-EditX TTS adapter.

Supports two modes:

- ``clone``: reference audio + reference text + target text → cloned speech
- ``edit``: reference audio + reference text + edit_type → transformed speech

Clone mode is used when no ``edit_type`` is provided. Edit mode triggers when
``edit_type`` is set.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .._adapter import TTSOutput


class StepAudioAdapter:
    def __init__(self, runtime):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> StepAudioAdapter:
        from ...generation.step_audio_editx import StepAudioEditXModel

        runtime = StepAudioEditXModel.from_dir(model_dir, prefer_mlx_int8=True)
        return cls(runtime)

    def generate(
        self,
        text: str | None = None,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        edit_type: str | None = None,
        edit_info: str | None = None,
        **kwargs,
    ) -> TTSOutput:
        if reference_audio is None or reference_text is None:
            raise ValueError(
                "Step Audio requires both reference_audio and reference_text"
            )

        from ...audio import load_audio

        if isinstance(reference_audio, mx.array):
            audio_np = np.asarray(reference_audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            sample_rate = 16000
        else:
            waveform, sample_rate = load_audio(str(reference_audio), mono=True)
            audio_np = np.asarray(waveform, dtype=np.float32)

        if edit_type is not None:
            result = self._runtime.edit(
                audio_np,
                int(sample_rate),
                reference_text,
                edit_type,
                edit_info=edit_info,
                target_text=text,
                max_new_tokens=max_new_tokens,
            )
        else:
            if text is None:
                raise ValueError(
                    "Step Audio clone mode requires text (target speech content)"
                )
            result = self._runtime.clone(
                audio_np,
                int(sample_rate),
                reference_text,
                text,
                max_new_tokens=max_new_tokens,
            )

        return TTSOutput(
            waveform=mx.array(result.waveform),
            sample_rate=result.sample_rate,
        )
