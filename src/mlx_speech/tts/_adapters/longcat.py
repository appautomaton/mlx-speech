"""LongCat AudioDiT TTS adapter."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.longcat_audiodit import synthesize_longcat_audiodit
from ...models.longcat_audiodit.checkpoint import (
    load_longcat_model,
    resolve_longcat_tokenizer_dir,
)
from ...models.longcat_audiodit.tokenizer import LongCatTokenizer


class LongCatAdapter:
    def __init__(self, model, tokenizer: LongCatTokenizer, sample_rate: int):
        self._model = model
        self._tokenizer = tokenizer
        self._sample_rate = sample_rate

    @classmethod
    def from_dir(cls, model_dir: Path) -> LongCatAdapter:
        loaded = load_longcat_model(model_dir)
        tok_dir = resolve_longcat_tokenizer_dir(model_dir=loaded.model_dir)
        tokenizer = LongCatTokenizer.from_path(tok_dir)
        return cls(loaded.model, tokenizer, loaded.config.sampling_rate)

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        max_new_tokens: int | None = None,
        nfe: int = 16,
        guidance_strength: float = 4.0,
        **kwargs,
    ) -> TTSOutput:
        prompt_audio = None
        if reference_audio is not None:
            if isinstance(reference_audio, (str, Path)):
                from ...audio import load_audio

                waveform, _ = load_audio(
                    reference_audio, sample_rate=self._sample_rate, mono=True
                )
                prompt_audio = waveform[None, None, :]
            else:
                prompt_audio = reference_audio

        result = synthesize_longcat_audiodit(
            model=self._model,
            tokenizer=self._tokenizer,
            text=text,
            prompt_text=reference_text,
            prompt_audio=prompt_audio,
            nfe=nfe,
            guidance_strength=guidance_strength,
        )
        return TTSOutput(waveform=result.waveform, sample_rate=result.sample_rate)
