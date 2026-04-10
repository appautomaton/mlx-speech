"""VibeVoice TTS adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.vibevoice import VibeVoiceGenerationConfig, synthesize_vibevoice
from ...models.vibevoice.checkpoint import load_vibevoice_model
from ...models.vibevoice.tokenizer import VibeVoiceTokenizer


class VibeVoiceAdapter:
    def __init__(self, model, tokenizer: VibeVoiceTokenizer, sample_rate: int):
        self._model = model
        self._tokenizer = tokenizer
        self._sample_rate = sample_rate

    @classmethod
    def from_dir(cls, model_dir: Path) -> VibeVoiceAdapter:
        loaded = load_vibevoice_model(model_dir)
        tokenizer = VibeVoiceTokenizer.from_path(model_dir)
        return cls(loaded.model, tokenizer, loaded.config.sampling_rate)

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        cfg_scale: float = 1.3,
        diffusion_steps: int = 20,
        **kwargs,
    ) -> TTSOutput:
        ref_mx = None
        if reference_audio is not None:
            if isinstance(reference_audio, (str, Path)):
                from ...audio import load_audio

                waveform, _ = load_audio(
                    reference_audio, sample_rate=self._sample_rate, mono=True
                )
                ref_mx = waveform[None, None, :]
            else:
                ref_mx = reference_audio

        config = VibeVoiceGenerationConfig(
            max_new_tokens=max_new_tokens or 2048,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
        )
        result = synthesize_vibevoice(
            self._model,
            self._tokenizer,
            text,
            reference_audio=ref_mx,
            config=config,
        )
        return TTSOutput(waveform=result.waveform, sample_rate=result.sample_rate)
