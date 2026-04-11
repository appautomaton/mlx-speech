"""MOSS TTS Delay adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.moss_delay import (
    MossTTSDelayGenerationConfig,
    synthesize_moss_tts_delay_conversations,
)
from ...models.moss_audio_tokenizer.checkpoint import load_moss_audio_tokenizer_model
from ...models.moss_delay.checkpoint import load_moss_tts_delay_model
from ...models.moss_delay.processor import MossTTSDelayProcessor


class MossDelayAdapter:
    def __init__(self, model, processor, sample_rate: int):
        self._model = model
        self._processor = processor
        self._sample_rate = sample_rate

    @classmethod
    def from_dir(cls, model_dir: Path, *, codec_dir: Path) -> MossDelayAdapter:
        loaded_model = load_moss_tts_delay_model(model_dir)
        loaded_codec = load_moss_audio_tokenizer_model(codec_dir)
        processor = MossTTSDelayProcessor.from_path(
            loaded_model.model_dir,
            audio_tokenizer=loaded_codec.model,
        )
        return cls(
            loaded_model.model,
            processor,
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
        config = MossTTSDelayGenerationConfig(
            **({"max_new_tokens": max_new_tokens} if max_new_tokens else {}),
        )
        # Build a proper TTSD user message via the processor. Passing a raw
        # `{"role": "user", "content": text}` dict would drop the text because
        # the processor reads `message["text"]`, not `message["content"]`.
        user_message = self._processor.build_user_message(text=text)
        batch_result = synthesize_moss_tts_delay_conversations(
            self._model,
            self._processor,
            conversations=[[user_message]],
            mode="generation",
            config=config,
        )
        output = batch_result.outputs[0]
        return TTSOutput(waveform=output.waveform, sample_rate=output.sample_rate)
