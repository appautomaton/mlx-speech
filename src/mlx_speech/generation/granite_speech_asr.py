"""Greedy inference for Granite Speech ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..models.granite_speech_asr import (
    GraniteSpeechConfig,
    GraniteSpeechFeatureExtractor,
    GraniteSpeechModel,
    GraniteSpeechTokenizer,
    greedy_next_token,
    mask_audio_token_ids,
    replace_audio_embeddings,
)


@dataclass(frozen=True)
class GraniteSpeechAsrResult:
    text: str
    tokens: list[int]
    language: str
    prompt_tokens: int


def validate_context_window(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    max_position_embeddings: int,
) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    requested = prompt_tokens + max_new_tokens
    if requested > max_position_embeddings:
        raise ValueError(
            f"Granite Speech request exceeds context: "
            f"prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, "
            f"max_position_embeddings={max_position_embeddings}."
        )


def _audio_to_numpy(
    audio: np.ndarray | mx.array | str | Path,
    *,
    sample_rate: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    if isinstance(audio, (str, Path)):
        from ..audio import load_audio

        waveform, loaded_sample_rate = load_audio(
            audio,
            sample_rate=target_sample_rate,
            mono=True,
        )
        return np.array(waveform, dtype=np.float32), loaded_sample_rate

    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Granite Speech requires {target_sample_rate} Hz audio; got {sample_rate} Hz. "
            "Resample before calling transcribe()."
        )

    if isinstance(audio, mx.array):
        return np.array(audio, dtype=np.float32), sample_rate
    return np.asarray(audio, dtype=np.float32), sample_rate


def _first_token_id(token: mx.array) -> int:
    return int(np.array(token).reshape(-1)[0])


@dataclass
class GraniteSpeechAsrModel:
    """Loaded Granite Speech model ready for local transcription."""

    model: GraniteSpeechModel
    feature_extractor: GraniteSpeechFeatureExtractor
    tokenizer: GraniteSpeechTokenizer
    config: GraniteSpeechConfig

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "GraniteSpeechAsrModel":
        bundle = GraniteSpeechModel.from_dir(model_dir, dtype=dtype, strict=True)
        return cls(
            model=bundle.model,
            feature_extractor=bundle.feature_extractor,
            tokenizer=bundle.tokenizer,
            config=bundle.config,
        )

    @classmethod
    def from_path(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "GraniteSpeechAsrModel":
        return cls.from_dir(model_dir, dtype=dtype)

    def transcribe(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str = "en",
        prompt: str | None = None,
        max_new_tokens: int = 448,
    ) -> GraniteSpeechAsrResult:
        target_sample_rate = int(self.feature_extractor.sample_rate)
        audio_np, sample_rate = _audio_to_numpy(
            audio,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
        )
        if sample_rate != target_sample_rate:
            raise ValueError(
                f"Granite Speech requires {target_sample_rate} Hz audio; got {sample_rate} Hz."
            )

        audio_shape = self.feature_extractor.preflight_shape(int(audio_np.shape[0]))
        num_audio_tokens = audio_shape.audio_tokens
        prompt_ids = self.tokenizer.build_prompt_ids(num_audio_tokens, prompt)
        validate_context_window(
            prompt_tokens=len(prompt_ids),
            max_new_tokens=max_new_tokens,
            max_position_embeddings=self.config.text.max_position_embeddings,
        )

        features_np, extracted_audio_tokens = self.feature_extractor(audio_np)
        if extracted_audio_tokens != num_audio_tokens:
            raise RuntimeError(
                f"Granite Speech audio token preflight mismatch: "
                f"preflight={num_audio_tokens}, extracted={extracted_audio_tokens}"
            )
        input_features = mx.array(features_np, dtype=mx.float32)
        audio_features = self.model.get_audio_features(input_features)
        mx.eval(audio_features)

        input_ids = mx.array([prompt_ids], dtype=mx.int32)
        masked_ids = mask_audio_token_ids(
            input_ids,
            audio_token_id=self.config.audio_token_index,
            replacement_id=0,
        )
        token_embeddings = self.model.embed_input_ids(masked_ids)
        inputs_embeds = replace_audio_embeddings(
            input_ids,
            token_embeddings,
            audio_features,
            audio_token_id=self.config.audio_token_index,
        )

        max_cache_len = len(prompt_ids) + max_new_tokens
        prefill = self.model.language_model.prefill(
            inputs_embeds=inputs_embeds,
            max_cache_len=max_cache_len,
        )
        mx.eval(prefill.logits)
        if prefill.kv_cache is None:
            raise RuntimeError("Granite Speech prefill did not return a KV cache")

        next_token = _first_token_id(greedy_next_token(prefill.logits))
        generated: list[int] = []
        eos_token_id = self.config.text.eos_token_id

        for idx in range(max_new_tokens):
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            if idx == max_new_tokens - 1:
                break
            step = self.model.language_model.decode_step(
                input_ids=mx.array([[next_token]], dtype=mx.int32),
                kv_cache=prefill.kv_cache,
            )
            mx.eval(step.logits)
            next_token = _first_token_id(greedy_next_token(step.logits))

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GraniteSpeechAsrResult(
            text=text,
            tokens=generated,
            language=language,
            prompt_tokens=len(prompt_ids),
        )

    def generate(self, audio: np.ndarray | mx.array | str | Path, **kwargs: Any) -> GraniteSpeechAsrResult:
        return self.transcribe(audio, **kwargs)
