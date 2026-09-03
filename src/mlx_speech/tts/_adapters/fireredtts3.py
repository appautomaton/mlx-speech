"""FireRedTTS3 Base adapter for the unified TTS API."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...audio import load_audio
from ...models.fireredtts3.core import FireRedTTS3Core
from ...models.fireredtts3.redae import RedAE
from ...models.fireredtts3.speaker import (
    FireRedSpeakerEncoder,
    resample_mono_audio,
)
from ...models.fireredtts3.tokenizer import FireRedTTS3Tokenizer
from .._adapter import TTSOutput


class FireRedTTS3Adapter:
    def __init__(
        self,
        *,
        core: FireRedTTS3Core,
        redae: RedAE,
        speaker: FireRedSpeakerEncoder,
        tokenizer: FireRedTTS3Tokenizer,
    ):
        self.core = core
        self.redae = redae
        self.speaker = speaker
        self.tokenizer = tokenizer

    @classmethod
    def from_dir(cls, model_dir: Path) -> "FireRedTTS3Adapter":
        return cls(
            core=FireRedTTS3Core.from_dir(model_dir),
            redae=RedAE.from_dir(model_dir),
            speaker=FireRedSpeakerEncoder.from_dir(model_dir),
            tokenizer=FireRedTTS3Tokenizer.from_dir(model_dir),
        )

    @property
    def sample_rate(self) -> int:
        return self.redae.sample_rate

    @staticmethod
    def _patch_budget(
        max_new_tokens: int | None,
        max_audio_patches: int | None,
    ) -> int:
        if (
            max_new_tokens is not None
            and max_audio_patches is not None
            and max_new_tokens != max_audio_patches
        ):
            raise ValueError(
                "max_new_tokens and max_audio_patches must match when both are set"
            )
        budget = max_audio_patches if max_audio_patches is not None else max_new_tokens
        if budget is None:
            return 400
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
            raise ValueError("FireRedTTS3 patch budget must be a positive integer")
        return int(budget)

    def _load_reference(
        self,
        reference_audio: str | Path | mx.array,
        *,
        reference_sample_rate: int | None,
    ) -> mx.array:
        if isinstance(reference_audio, (str, Path)):
            waveform, source_rate = load_audio(reference_audio, mono=True)
        else:
            waveform = reference_audio
            source_rate = (
                self.sample_rate
                if reference_sample_rate is None
                else int(reference_sample_rate)
            )
        return resample_mono_audio(
            waveform,
            source_rate=source_rate,
            target_rate=self.sample_rate,
        )

    def generate(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        language: str = "English",
        seed: int | None = 1234,
        guidance_scale: float = 2.0,
        flow_steps: int = 10,
        stop_threshold: float = 0.5,
        max_new_tokens: int | None = None,
        max_audio_patches: int | None = None,
        **kwargs,
    ) -> TTSOutput:
        del kwargs
        if not isinstance(text, str) or not text.strip():
            raise ValueError("FireRedTTS3 text must not be empty")
        if reference_audio is None or reference_text is None:
            raise ValueError(
                "FireRedTTS3 requires reference_audio and reference_text together"
            )
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("FireRedTTS3 reference_text must not be empty")
        if not np.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError(
                "FireRedTTS3 guidance_scale must be finite and non-negative"
            )
        if not np.isfinite(stop_threshold) or not 0 <= stop_threshold <= 1:
            raise ValueError("FireRedTTS3 stop_threshold must be between 0 and 1")
        patch_budget = self._patch_budget(max_new_tokens, max_audio_patches)

        waveform = self._load_reference(
            reference_audio,
            reference_sample_rate=reference_sample_rate,
        )
        multiple = self.redae.downsample_rate * self.core.config.patch_size
        prompt_audio = self.redae.pad_audio(waveform[None], multiple=multiple)
        prompt_samples = int(prompt_audio.shape[-1])
        prompt_latents = self.redae.encode(prompt_audio[0]).astype(mx.float32)
        speaker_embedding = self.speaker(
            prompt_audio[0],
            sample_rate=self.sample_rate,
        )
        mx.eval(prompt_latents, speaker_embedding)

        token_ids = self.tokenizer.encode(
            language=language,
            reference_text=reference_text,
            text=text,
        )
        result = self.core.generate(
            speaker_embedding=speaker_embedding,
            text_tokens=mx.array([token_ids], dtype=mx.int32),
            prompt_latents=prompt_latents,
            flow_steps=flow_steps,
            guidance_scale=guidance_scale,
            stop_threshold=stop_threshold,
            max_generated_patches=patch_budget,
            seed=seed,
        )
        decoded = self.redae.decode(result.latents).astype(mx.float32)
        waveform = decoded[0, prompt_samples:]
        mx.eval(waveform)
        if int(waveform.size) == 0:
            raise RuntimeError("FireRedTTS3 generated an empty waveform")
        if not bool(mx.all(mx.isfinite(waveform)).item()):
            raise RuntimeError("FireRedTTS3 generated non-finite waveform samples")
        return TTSOutput(waveform=waveform, sample_rate=self.sample_rate)


__all__ = ["FireRedTTS3Adapter"]
