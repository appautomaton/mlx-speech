"""Breeze TTS 2 adapter for the unified batch and streaming TTS APIs."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import mlx.core as mx

from ...audio import load_audio, mix_down_mono, resample_audio
from ...generation.breeze_tts import (
    SAMPLE_RATE,
    generate_breeze,
    generate_breeze_stream,
    load_breeze_speech,
)
from .._adapter import TTSOutput


class BreezeTTSAdapter:
    def __init__(self, speech, codec, tokenizer) -> None:
        self._speech = speech
        self._codec = codec
        self._tokenizer = tokenizer

    @classmethod
    def from_dir(cls, model_dir: Path) -> "BreezeTTSAdapter":
        speech, codec, tokenizer = load_breeze_speech(model_dir)
        return cls(speech, codec, tokenizer)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def generate(
        self,
        text: str | None = None,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        max_new_tokens: int | None = None,
        max_frames: int | None = None,
        instruction: str | None = None,
        cfg_scale: float | None = None,
        guidance_scale: float | None = None,
        speaker: str = "S0",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 0,
        **kwargs,
    ) -> TTSOutput:
        del kwargs
        audio = generate_breeze(
            self._speech,
            self._codec,
            self._tokenizer,
            self._text(text),
            **self._controls(
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sample_rate=reference_sample_rate,
                max_new_tokens=max_new_tokens,
                max_frames=max_frames,
                instruction=instruction,
                cfg_scale=cfg_scale,
                guidance_scale=guidance_scale,
                speaker=speaker,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
        )
        return TTSOutput(waveform=audio.samples, sample_rate=audio.sample_rate)

    def generate_stream(
        self,
        text: str | None = None,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        max_new_tokens: int | None = None,
        max_frames: int | None = None,
        instruction: str | None = None,
        cfg_scale: float | None = None,
        guidance_scale: float | None = None,
        speaker: str = "S0",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 0,
        stream_chunk_patches: int = 4,
        **kwargs,
    ):
        del kwargs
        chunks = generate_breeze_stream(
            self._speech,
            self._codec,
            self._tokenizer,
            self._text(text),
            chunk_frames=self._chunk_frames(stream_chunk_patches),
            **self._controls(
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sample_rate=reference_sample_rate,
                max_new_tokens=max_new_tokens,
                max_frames=max_frames,
                instruction=instruction,
                cfg_scale=cfg_scale,
                guidance_scale=guidance_scale,
                speaker=speaker,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
        )
        for chunk in chunks:
            yield TTSOutput(waveform=chunk, sample_rate=SAMPLE_RATE)

    def _controls(
        self,
        *,
        reference_audio: str | Path | mx.array | None,
        reference_text: str | None,
        reference_sample_rate: int | None,
        max_new_tokens: int | None,
        max_frames: int | None,
        instruction: str | None,
        cfg_scale: float | None,
        guidance_scale: float | None,
        speaker: str,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
    ) -> dict:
        return {
            "speaker": speaker,
            "instruction": instruction,
            "cfg_scale": self._cfg_scale(cfg_scale, guidance_scale),
            "max_frames": self._frame_budget(max_new_tokens, max_frames),
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "ref_audio": self._reference_waveform(
                reference_audio, reference_sample_rate
            ),
            "ref_text": reference_text,
        }

    @staticmethod
    def _text(text: str | None) -> str:
        if text is None or not str(text).strip():
            raise ValueError("Breeze TTS requires text.")
        return text

    @staticmethod
    def _cfg_scale(cfg_scale: float | None, guidance_scale: float | None) -> float:
        if (
            cfg_scale is not None
            and guidance_scale is not None
            and cfg_scale != guidance_scale
        ):
            raise ValueError(
                "cfg_scale and guidance_scale must match when both are set."
            )
        value = cfg_scale if cfg_scale is not None else guidance_scale
        return 1.0 if value is None else float(value)

    @staticmethod
    def _frame_budget(max_new_tokens: int | None, max_frames: int | None) -> int:
        if (
            max_new_tokens is not None
            and max_frames is not None
            and max_new_tokens != max_frames
        ):
            raise ValueError(
                "max_new_tokens and max_frames must match when both are set."
            )
        budget = max_frames if max_frames is not None else max_new_tokens
        if budget is None:
            return 80
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
            raise ValueError("Breeze frame budget must be a positive integer.")
        return int(budget)

    @staticmethod
    def _chunk_frames(stream_chunk_patches: int) -> int:
        if (
            isinstance(stream_chunk_patches, bool)
            or not isinstance(stream_chunk_patches, Integral)
            or stream_chunk_patches <= 0
        ):
            raise ValueError("stream_chunk_patches must be a positive integer.")
        return int(stream_chunk_patches)

    @staticmethod
    def _reference_waveform(
        reference_audio: str | Path | mx.array | None,
        reference_sample_rate: int | None,
    ) -> mx.array | None:
        if reference_audio is None:
            return None
        if isinstance(reference_audio, (str, Path)):
            waveform, _rate = load_audio(
                reference_audio, sample_rate=SAMPLE_RATE, mono=True
            )
            return waveform
        if not isinstance(reference_audio, mx.array):
            raise TypeError("reference_audio must be a path or an MLX array.")
        waveform = (
            reference_audio
            if reference_audio.ndim == 1
            else mix_down_mono(reference_audio)
        )
        rate = (
            SAMPLE_RATE if reference_sample_rate is None else int(reference_sample_rate)
        )
        if rate != SAMPLE_RATE:
            waveform = resample_audio(
                waveform,
                orig_sample_rate=rate,
                target_sample_rate=SAMPLE_RATE,
            )
        return waveform


__all__ = ["BreezeTTSAdapter"]
