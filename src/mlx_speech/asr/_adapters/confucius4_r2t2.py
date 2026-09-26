"""Confucius4-R2T2 streaming adapter.

Offline ``generate`` is the shared Qwen3-ASR one-shot path. ``stream_session``
adds NetEase's chunk loop (``models/confucius4_r2t2/streaming.py``) on top of a
per-step greedy decode of ``prompt + prefix`` over all audio seen so far.

The state machine owns chunking, prefix rollback, the token budget, and the
append-only commit. This module only supplies the two things it cannot know:
how to build ``prompt + prefix`` for the current audio, and how to run one
short greedy decode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...generation.qwen3_asr import (
    Qwen3ASRTranscriber,
    greedy_next_token,
    validate_context_window,
)
from ...models.confucius4_r2t2.streaming import (
    R2T2StreamSession,
    R2T2StreamUpdate,
)
from .._adapter import ASROutput

SAMPLE_RATE = 16_000

StepDecoder = Callable[[str, np.ndarray, int], str]


class Confucius4R2T2Adapter:
    """Qwen3-ASR graph with NetEase's R2T2 finetune and its streaming loop."""

    def __init__(
        self,
        runtime: Qwen3ASRTranscriber,
        *,
        step_decoder: StepDecoder | None = None,
    ) -> None:
        self._runtime = runtime
        self._step_decoder = step_decoder

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Confucius4R2T2Adapter":
        return cls(Qwen3ASRTranscriber.from_dir(model_dir))

    def generate(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        **kwargs,
    ) -> ASROutput:
        result = self._runtime.transcribe(
            audio,
            sample_rate=sample_rate,
            language=language,
            **kwargs,
        )
        return ASROutput(text=result.text, language=result.language)

    def stream_session(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        context: str = "",
        chunk_ms: int = 160,
        lookahead_ms: int = 160,
        unfixed_token_num: int = 1,
        unfixed_chunk_num: int = 0,
        rollback_punctuation: bool = False,
    ) -> "Confucius4R2T2StreamSession":
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Confucius4-R2T2 streaming requires {SAMPLE_RATE} Hz; "
                f"got {sample_rate}"
            )
        tokenizer = self._runtime.processor.tokenizer
        decoder = self._step_decoder or self._build_step_decoder(
            context=context,
            language=language,
        )
        state = R2T2StreamSession(
            decoder,
            tokenizer.encode,
            lambda token_ids: tokenizer.decode(list(token_ids)),
            language=language,
            chunk_ms=chunk_ms,
            lookahead_ms=lookahead_ms,
            unfixed_token_num=unfixed_token_num,
            unfixed_chunk_num=unfixed_chunk_num,
            rollback_punctuation=rollback_punctuation,
        )
        return Confucius4R2T2StreamSession(state)

    def _build_step_decoder(self, *, context: str, language: str | None) -> StepDecoder:
        runtime = self._runtime

        def decode(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
            return _completion(
                runtime,
                prompt_suffix=prefix,
                audio=audio,
                max_new_tokens=max_new_tokens,
                context=context,
                language=language,
            )

        return decode


class Confucius4R2T2StreamSession:
    """Adapter view of the state machine: ``feed`` frames, then ``finalize``."""

    def __init__(self, state: R2T2StreamSession) -> None:
        self._state = state

    @property
    def language(self) -> str | None:
        return self._state.language

    @property
    def audio_samples(self) -> int:
        return self._state.audio_samples

    @property
    def chunk_id(self) -> int:
        return self._state.chunk_id

    def feed(self, pcm16k: np.ndarray) -> R2T2StreamUpdate:
        return self._state.feed(pcm16k)

    def finalize(self) -> ASROutput:
        transcript = self._state.finalize()
        return ASROutput(text=transcript.text, language=transcript.language)


def _completion(
    runtime: Qwen3ASRTranscriber,
    *,
    prompt_suffix: str,
    audio: np.ndarray,
    max_new_tokens: int,
    context: str,
    language: str | None,
) -> str:
    """One greedy decode of ``prompt + prefix`` over the accumulated audio."""

    processor = runtime.processor
    model = runtime.model
    feature_batch = processor.feature_extractor(audio, sample_rate=SAMPLE_RATE)
    if int(feature_batch.input_features.shape[0]) != 1:
        raise ValueError(
            "Confucius4-R2T2 streaming supports one audio input at a time."
        )

    audio_length = int(feature_batch.audio_lengths.reshape(-1)[0])
    prompt_text = _prompt_text(
        processor,
        prompt_suffix=prompt_suffix,
        audio_length=audio_length,
        context=context,
        language=language,
    )
    input_ids = processor.tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
    )
    validate_context_window(
        prompt_tokens=len(input_ids),
        max_new_tokens=max_new_tokens,
        max_position_embeddings=runtime.config.text_config.max_position_embeddings,
    )

    audio_features = model.get_audio_features(
        mx.array(feature_batch.input_features, dtype=mx.float32),
        feature_attention_mask=mx.array(
            feature_batch.feature_attention_mask,
            dtype=mx.int32,
        ),
    )
    mx.eval(audio_features)

    inputs_embeds = model.prepare_inputs_embeds(
        mx.array([input_ids], dtype=mx.int32),
        audio_features,
    )
    prefill = model.prefill(
        inputs_embeds=inputs_embeds,
        max_cache_len=len(input_ids) + max_new_tokens,
    )
    mx.eval(prefill.logits)
    if prefill.past_key_values is None:
        raise RuntimeError("Confucius4-R2T2 prefill did not return a KV cache.")

    generated = _greedy_tokens(
        model,
        prefill,
        max_new_tokens=max_new_tokens,
        eos_token_ids=runtime.eos_token_ids,
    )
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def _prompt_text(
    processor,
    *,
    prompt_suffix: str,
    audio_length: int,
    context: str,
    language: str | None,
) -> str:
    """The processor template with the prefix after the language suffix."""

    prompt = processor.build_prompt(
        context=context,
        audio_length=audio_length,
        language=language,
    )
    return prompt.prompt + prompt_suffix


def _greedy_tokens(
    model,
    prefill,
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
) -> list[int]:
    """Decode only the completion; the prompt prefix is already in the cache."""

    eos = set(eos_token_ids)
    next_token = _token_id(greedy_next_token(prefill.logits))
    generated: list[int] = []
    for index in range(max_new_tokens):
        if next_token in eos:
            break
        generated.append(next_token)
        if index == max_new_tokens - 1:
            break
        step = model.decode_step(
            input_ids=mx.array([[next_token]], dtype=mx.int32),
            kv_cache=prefill.past_key_values,
        )
        mx.eval(step.logits)
        next_token = _token_id(greedy_next_token(step.logits))
    return generated


def _token_id(token: mx.array) -> int:
    return int(np.array(token).reshape(-1)[0])


__all__ = ["Confucius4R2T2Adapter", "Confucius4R2T2StreamSession"]
