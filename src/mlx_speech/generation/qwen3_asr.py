"""Greedy local inference for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..models.qwen3_asr.config import Qwen3ASRConfig
from ..models.qwen3_asr.model import Qwen3ASRModel, replace_audio_embeddings
from ..models.qwen3_asr.processor import Qwen3ASRProcessor, parse_asr_output


@dataclass(frozen=True)
class Qwen3ASRResult:
    text: str
    tokens: list[int]
    language: str
    raw_text: str
    prompt_tokens: int


def validate_context_window(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    max_position_embeddings: int,
) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    requested = int(prompt_tokens) + int(max_new_tokens)
    if requested > int(max_position_embeddings):
        raise ValueError(
            "Qwen3-ASR request exceeds context: "
            f"prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, "
            f"max_position_embeddings={max_position_embeddings}."
        )


def greedy_next_token(logits: mx.array) -> mx.array:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, vocab], got {logits.shape}.")
    return mx.argmax(logits[:, -1, :], axis=-1)


@dataclass
class Qwen3ASRTranscriber:
    model: Qwen3ASRModel
    processor: Qwen3ASRProcessor
    config: Qwen3ASRConfig

    def transcribe(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        context: str = "",
        max_new_tokens: int = 448,
    ) -> Qwen3ASRResult:
        prompt = None
        sample_count = _audio_sample_count(audio)
        if sample_count is not None:
            _, preflight_audio_tokens = self.processor.feature_extractor.preflight_shape(sample_count)
            prompt = self.processor.build_prompt(
                context=context,
                audio_length=preflight_audio_tokens,
                language=language,
            )
            validate_context_window(
                prompt_tokens=len(prompt.input_ids),
                max_new_tokens=max_new_tokens,
                max_position_embeddings=self.config.text_config.max_position_embeddings,
            )

        feature_batch = self.processor.feature_extractor(audio, sample_rate=sample_rate)
        if int(feature_batch.input_features.shape[0]) != 1:
            raise ValueError("Qwen3-ASR v0 transcribe() supports one audio input at a time.")

        audio_length = int(feature_batch.audio_lengths.reshape(-1)[0])
        if prompt is None:
            prompt = self.processor.build_prompt(
                context=context,
                audio_length=audio_length,
                language=language,
            )
            validate_context_window(
                prompt_tokens=len(prompt.input_ids),
                max_new_tokens=max_new_tokens,
                max_position_embeddings=self.config.text_config.max_position_embeddings,
            )
        elif prompt.audio_length != audio_length:
            raise RuntimeError(
                f"Qwen3-ASR audio token preflight mismatch: "
                f"preflight={prompt.audio_length}, extracted={audio_length}."
            )

        input_features = mx.array(feature_batch.input_features, dtype=mx.float32)
        feature_attention_mask = mx.array(feature_batch.feature_attention_mask, dtype=mx.int32)
        audio_features = self.model.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
        )
        mx.eval(audio_features)

        input_ids = mx.array([prompt.input_ids], dtype=mx.int32)
        token_embeddings = self.model.embed_input_ids(input_ids)
        inputs_embeds = replace_audio_embeddings(
            input_ids,
            token_embeddings,
            audio_features,
            audio_token_id=self.config.audio_token_id,
        )

        max_cache_len = len(prompt.input_ids) + max_new_tokens
        prefill = self.model.prefill(
            inputs_embeds=inputs_embeds,
            max_cache_len=max_cache_len,
        )
        mx.eval(prefill.logits)
        if prefill.past_key_values is None:
            raise RuntimeError("Qwen3-ASR prefill did not return a KV cache.")

        next_token = _first_token_id(greedy_next_token(prefill.logits))
        generated: list[int] = []
        eos_token_ids = _eos_token_ids(self.config.text_config.eos_token_id)
        for index in range(max_new_tokens):
            if next_token in eos_token_ids:
                break
            generated.append(next_token)
            if index == max_new_tokens - 1:
                break
            step = self.model.decode_step(
                input_ids=mx.array([[next_token]], dtype=mx.int32),
                kv_cache=prefill.past_key_values,
            )
            mx.eval(step.logits)
            next_token = _first_token_id(greedy_next_token(step.logits))

        raw_text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        detected_language, text = parse_asr_output(raw_text, user_language=prompt.language)
        return Qwen3ASRResult(
            text=text,
            tokens=generated,
            language=detected_language,
            raw_text=raw_text,
            prompt_tokens=len(prompt.input_ids),
        )

    def generate(self, audio: np.ndarray | mx.array | str | Path, **kwargs: Any) -> Qwen3ASRResult:
        return self.transcribe(audio, **kwargs)


def _audio_sample_count(audio: np.ndarray | mx.array | str | Path) -> int | None:
    if isinstance(audio, (str, Path)):
        return None
    return int(audio.shape[0]) if getattr(audio, "ndim", 0) == 1 else None


def _first_token_id(token: mx.array) -> int:
    return int(np.array(token).reshape(-1)[0])


def _eos_token_ids(eos_token_id: int | list[int] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, list):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}
