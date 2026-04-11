"""MOSS Sound Effect TTS adapter — text-to-sound-effect generation."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput


class MossSoundEffectAdapter:
    def __init__(self, model, processor, sample_rate: int):
        self._model = model
        self._processor = processor
        self._sample_rate = sample_rate

    @classmethod
    def from_dir(cls, model_dir: Path, *, codec_dir: Path) -> MossSoundEffectAdapter:
        from ...models.moss_audio_tokenizer.checkpoint import (
            load_moss_audio_tokenizer_model,
        )
        from ...models.moss_delay import load_moss_sound_effect_model
        from ...models.moss_delay.processor import MossTTSDelayProcessor

        loaded = load_moss_sound_effect_model(model_dir)
        loaded_codec = load_moss_audio_tokenizer_model(codec_dir)
        processor = MossTTSDelayProcessor.from_path(
            loaded.model_dir,
            audio_tokenizer=loaded_codec.model,
        )
        return cls(loaded.model, processor, loaded.config.sampling_rate)

    def generate(
        self,
        text: str | None = None,
        *,
        duration_seconds: float | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> TTSOutput:
        if text is None:
            raise ValueError("Moss Sound Effect requires text (ambient sound description)")

        from ...generation.moss_delay import (
            MossTTSDelayGenerationConfig,
            synthesize_moss_tts_delay_conversations,
        )
        from ...models.moss_delay.sound_effect import (
            SOUND_EFFECT_DEFAULT_AUDIO_REPETITION_PENALTY,
            SOUND_EFFECT_DEFAULT_AUDIO_TEMPERATURE,
            SOUND_EFFECT_DEFAULT_AUDIO_TOP_K,
            SOUND_EFFECT_DEFAULT_AUDIO_TOP_P,
            SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS,
            build_sound_effect_conversation,
        )

        conversations, _expected_tokens = build_sound_effect_conversation(
            self._processor,
            ambient_sound=text,
            duration_seconds=duration_seconds,
        )
        # Use sound-effect-specific defaults (temp=1.5, top_p=0.6, rep_pen=1.2,
        # max_new_tokens=4096), not the dialogue-TTSD defaults.
        # NOTE: expected_tokens is a prompt HINT baked into the conversation
        # (telling the model the target length); it is NOT a generation budget.
        # Use SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS so the model can actually
        # run to its natural stop.
        config = MossTTSDelayGenerationConfig(
            max_new_tokens=max_new_tokens or SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS,
            audio_temperature=SOUND_EFFECT_DEFAULT_AUDIO_TEMPERATURE,
            audio_top_p=SOUND_EFFECT_DEFAULT_AUDIO_TOP_P,
            audio_top_k=SOUND_EFFECT_DEFAULT_AUDIO_TOP_K,
            audio_repetition_penalty=SOUND_EFFECT_DEFAULT_AUDIO_REPETITION_PENALTY,
        )
        batch = synthesize_moss_tts_delay_conversations(
            self._model,
            self._processor,
            conversations=conversations,
            mode="generation",
            config=config,
        )
        output = batch.outputs[0]
        return TTSOutput(waveform=output.waveform, sample_rate=output.sample_rate)
