"""MOSS TTS Local adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.moss_local import (
    MossTTSLocalGenerationConfig,
    synthesize_moss_tts_local,
)
from ...models.moss_audio_tokenizer.checkpoint import load_moss_audio_tokenizer_model
from ...models.moss_local.checkpoint import load_moss_tts_local_model
from ...models.moss_local.processor import MossTTSLocalProcessor


class MossLocalAdapter:
    def __init__(self, model, processor, codec, sample_rate: int):
        self._model = model
        self._processor = processor
        self._codec = codec
        self._sample_rate = sample_rate

    @classmethod
    def from_dir(cls, model_dir: Path, *, codec_dir: Path) -> MossLocalAdapter:
        loaded_model = load_moss_tts_local_model(model_dir)
        loaded_codec = load_moss_audio_tokenizer_model(codec_dir)
        processor = MossTTSLocalProcessor.from_path(
            loaded_model.model_dir,
            audio_tokenizer=loaded_codec.model,
        )
        return cls(
            loaded_model.model,
            processor,
            loaded_codec.model,
            loaded_model.config.sampling_rate,
        )

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> TTSOutput:
        config = MossTTSLocalGenerationConfig.app_defaults(
            **({"max_new_tokens": max_new_tokens} if max_new_tokens else {}),
        )
        result = synthesize_moss_tts_local(
            self._model,
            self._processor,
            self._codec,
            text=text,
            config=config,
        )
        return TTSOutput(waveform=result.waveform, sample_rate=self._sample_rate)
