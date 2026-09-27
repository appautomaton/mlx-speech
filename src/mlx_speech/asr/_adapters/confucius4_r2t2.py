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
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...generation.qwen3_asr import (
    Qwen3ASRTranscriber,
    greedy_next_token,
    validate_context_window,
)
from ...models.qwen3_asr import _get_feat_extract_output_lengths
from ...models.qwen3_asr.audio_encoder import _attention_block_lengths
from ...models.qwen3_asr.text_decoder import (
    REFERENCE_DECODE_COMPUTE,
    Qwen3ASRDecodeCompute,
    Qwen3ASRTextCausalLMOutput,
    Qwen3ASRTextKVCache,
)
from ...models.confucius4_r2t2.metrics import (
    CACHE_ALLOCATIONS,
    DECODE_STEPS,
    ENCODER_BLOCKS,
    ENCODER_TOKENS,
    MEL_FRAMES,
    PREFILL_POSITIONS,
    VOCAB_POSITIONS,
    R2T2StepMetrics,
    R2T2StreamTrace,
    stage,
)
from ...models.confucius4_r2t2.streaming import (
    R2T2StreamSession,
    R2T2StreamUpdate,
)
from .._adapter import ASROutput
from .confucius4_r2t2_incremental import R2T2IncrementalStep

SAMPLE_RATE = 16_000
_TOP_K = 5

StepDecoder = Callable[[str, np.ndarray, int], str]
LogitsObserver = Callable[[int, mx.array], None]


@dataclass(frozen=True)
class R2T2StreamCompute:
    """Compute options for one streaming step. The defaults are the reference
    (precision-oracle) path; R2T2 semantics are identical under every option.

    ``decoder`` selects the decoder matmul and attention implementation.
    ``last_logits_only`` projects only the last prefill position to the
    vocabulary. ``audio_bf16`` feeds bf16 features to the audio tower.
    ``residual_bf16`` casts the prompt embeddings, and therefore the prefill
    residual stream and KV cache, to bf16. ``incremental`` keeps per-session
    mel, audio-block, and KV state and recomputes only what changed.
    """

    decoder: Qwen3ASRDecodeCompute = field(default=REFERENCE_DECODE_COMPUTE)
    last_logits_only: bool = False
    audio_bf16: bool = False
    residual_bf16: bool = False
    incremental: bool = False


REFERENCE_STREAM_COMPUTE = R2T2StreamCompute()
FAST_DECODE_COMPUTE = Qwen3ASRDecodeCompute(
    native_linear=True, native_lm_head=True, fused_attention=True
)
# Full-refeed with native bf16 compute: the oracle for incremental reuse.
FULL_REFEED_STREAM_COMPUTE = R2T2StreamCompute(
    decoder=FAST_DECODE_COMPUTE,
    last_logits_only=True,
    audio_bf16=True,
    residual_bf16=True,
)
DEFAULT_STREAM_COMPUTE = R2T2StreamCompute(
    decoder=FAST_DECODE_COMPUTE,
    last_logits_only=True,
    audio_bf16=True,
    residual_bf16=True,
    incremental=True,
)


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
        trace: R2T2StreamTrace | None = None,
        compute: R2T2StreamCompute | None = None,
    ) -> "Confucius4R2T2StreamSession":
        """Open a streaming session.

        ``trace`` is opt-in profiling. With the default ``None`` the session
        keeps no per-step records; the benchmark harness passes one.
        ``compute`` overrides the step compute options (benchmarks and parity
        gates); ``None`` uses ``DEFAULT_STREAM_COMPUTE``.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Confucius4-R2T2 streaming requires {SAMPLE_RATE} Hz; "
                f"got {sample_rate}"
            )
        tokenizer = self._runtime.processor.tokenizer
        decoder = self._step_decoder or self._build_step_decoder(
            context=context,
            language=language,
            trace=trace,
            compute=compute,
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
            trace=trace,
        )
        return Confucius4R2T2StreamSession(state)

    def _build_step_decoder(
        self,
        *,
        context: str,
        language: str | None,
        trace: R2T2StreamTrace | None = None,
        compute: R2T2StreamCompute | None = None,
    ) -> StepDecoder:
        runtime = self._runtime
        step_compute = DEFAULT_STREAM_COMPUTE if compute is None else compute

        if step_compute.incremental:
            step = R2T2IncrementalStep(
                runtime, context=context, language=language, compute=step_compute
            )

            def decode_incremental(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
                return _incremental_completion(
                    runtime,
                    step,
                    prompt_suffix=prefix,
                    audio=audio,
                    max_new_tokens=max_new_tokens,
                    context=context,
                    language=language,
                    metrics=trace.current if trace is not None else None,
                    compute=step_compute,
                )

            return decode_incremental

        def decode(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
            return _completion(
                runtime,
                prompt_suffix=prefix,
                audio=audio,
                max_new_tokens=max_new_tokens,
                context=context,
                language=language,
                metrics=trace.current if trace is not None else None,
                compute=step_compute,
            )

        return decode


class Confucius4R2T2StreamSession:
    """Adapter view of the state machine: ``feed`` frames, then ``finalize``."""

    def __init__(self, state: R2T2StreamSession) -> None:
        self._state = state

    @property
    def trace(self) -> R2T2StreamTrace | None:
        """Profiling records when the session was opened with a trace."""

        return self._state.trace

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
    metrics: R2T2StepMetrics | None = None,
    compute: R2T2StreamCompute | None = None,
    logits_observer: LogitsObserver | None = None,
) -> str:
    """One greedy decode of ``prompt + prefix`` over the accumulated audio.

    ``metrics`` is opt-in per-window instrumentation. With ``None`` the body
    runs no timers, no counters, and no extra ``mx.eval`` barriers.
    ``logits_observer`` receives ``(generation_index, last_logits)`` for every
    greedy choice; parity gates use it.
    """

    compute = REFERENCE_STREAM_COMPUTE if compute is None else compute
    processor = runtime.processor
    model = runtime.model
    with stage(metrics, "mel"):
        feature_batch = processor.feature_extractor(audio, sample_rate=SAMPLE_RATE)
    if int(feature_batch.input_features.shape[0]) != 1:
        raise ValueError(
            "Confucius4-R2T2 streaming supports one audio input at a time."
        )

    audio_length = int(feature_batch.audio_lengths.reshape(-1)[0])
    with stage(metrics, "prompt_build"):
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

    with stage(metrics, "audio_tower"):
        audio_features = model.get_audio_features(
            mx.array(
                feature_batch.input_features,
                dtype=mx.bfloat16 if compute.audio_bf16 else mx.float32,
            ),
            feature_attention_mask=mx.array(
                feature_batch.feature_attention_mask,
                dtype=mx.int32,
            ),
        )
        mx.eval(audio_features)

    with stage(metrics, "embed"):
        inputs_embeds = model.prepare_inputs_embeds(
            mx.array([input_ids], dtype=mx.int32),
            audio_features,
        )
        if compute.residual_bf16:
            inputs_embeds = inputs_embeds.astype(mx.bfloat16)

    max_cache_len = len(input_ids) + max_new_tokens
    if metrics is None:
        prefill = model.prefill(
            inputs_embeds=inputs_embeds,
            max_cache_len=max_cache_len,
            compute=compute.decoder,
            last_logits_only=compute.last_logits_only,
        )
        mx.eval(prefill.logits)
    else:
        prefill = _profiled_prefill(
            model,
            inputs_embeds=inputs_embeds,
            max_cache_len=max_cache_len,
            metrics=metrics,
            compute=compute,
        )
    if prefill.past_key_values is None:
        raise RuntimeError("Confucius4-R2T2 prefill did not return a KV cache.")

    if metrics is not None:
        feature_frames = int(np.asarray(feature_batch.feature_attention_mask)[0].sum())
        _record_window_work(
            model,
            metrics,
            mel_frames=feature_frames,
            audio_tokens=int(audio_features.shape[0]),
            prompt_tokens=len(input_ids),
            vocab_positions=1 if compute.last_logits_only else len(input_ids),
        )

    generated = _greedy_tokens(
        model,
        prefill,
        max_new_tokens=max_new_tokens,
        eos_token_ids=runtime.eos_token_ids,
        metrics=metrics,
        decoder_compute=compute.decoder,
        logits_observer=logits_observer,
    )
    completion = processor.tokenizer.decode(generated, skip_special_tokens=True)
    if metrics is not None:
        metrics.completion = completion
        metrics.generated_ids = tuple(generated)
    return completion


def _incremental_completion(
    runtime: Qwen3ASRTranscriber,
    step: R2T2IncrementalStep,
    *,
    prompt_suffix: str,
    audio: np.ndarray,
    max_new_tokens: int,
    context: str,
    language: str | None,
    metrics: R2T2StepMetrics | None = None,
    compute: R2T2StreamCompute,
    logits_observer: LogitsObserver | None = None,
) -> str:
    """``_completion`` with per-session reuse; same prompt and greedy decode."""

    processor = runtime.processor

    def prompt_ids(audio_tokens: int) -> list[int]:
        text = _prompt_text(
            processor,
            prompt_suffix=prompt_suffix,
            audio_length=audio_tokens,
            context=context,
            language=language,
        )
        return list(processor.tokenizer.encode(text, add_special_tokens=False))

    prefill = step.prefill(
        prompt_ids=prompt_ids,
        audio=audio,
        max_new_tokens=max_new_tokens,
        metrics=metrics,
    )
    if metrics is not None:
        _capture_logits(metrics, prefill.logits)
    generated = _greedy_tokens(
        runtime.model,
        prefill,
        max_new_tokens=max_new_tokens,
        eos_token_ids=runtime.eos_token_ids,
        metrics=metrics,
        decoder_compute=compute.decoder,
        logits_observer=logits_observer,
    )
    step.record_generated(generated)
    completion = processor.tokenizer.decode(generated, skip_special_tokens=True)
    if metrics is not None:
        metrics.completion = completion
        metrics.generated_ids = tuple(generated)
    return completion


def _profiled_prefill(
    model,
    *,
    inputs_embeds: mx.array,
    max_cache_len: int,
    metrics: R2T2StepMetrics,
    compute: R2T2StreamCompute,
) -> Qwen3ASRTextCausalLMOutput:
    """Prefill with the body and the vocabulary head timed apart.

    Profiling only. It runs the same two steps ``Qwen3ASRModel.prefill`` runs,
    with an ``mx.eval`` barrier between them so the harness can attribute cost.
    """

    text_decoder = model.text_decoder
    kv_cache = Qwen3ASRTextKVCache.allocate(
        model.config.text_config,
        batch_size=int(inputs_embeds.shape[0]),
        max_length=max_cache_len,
        dtype=inputs_embeds.dtype,
    )
    with stage(metrics, "prefill_body"):
        body = text_decoder.model.prefill(
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            compute=compute.decoder,
        )
        mx.eval(body.last_hidden_state)
    hidden = body.last_hidden_state
    if compute.last_logits_only:
        hidden = hidden[:, -1:, :]
    with stage(metrics, "vocab"):
        logits = text_decoder.project_logits(hidden, compute=compute.decoder)
        mx.eval(logits)
    _capture_logits(metrics, logits)
    return Qwen3ASRTextCausalLMOutput(
        logits=logits,
        last_hidden_state=body.last_hidden_state,
        past_key_values=kv_cache,
    )


def _capture_logits(
    metrics: R2T2StepMetrics,
    logits: mx.array,
    *,
    top_k: int = _TOP_K,
) -> None:
    """Record the top-k logits and the top-1/top-2 margin at the last position."""

    last = np.asarray(logits[0, -1, :], dtype=np.float32)
    order = np.argsort(-last)
    keep = min(int(top_k), int(order.shape[0]))
    metrics.top_logits = tuple(
        (int(order[position]), float(last[order[position]]))
        for position in range(keep)
    )
    if int(order.shape[0]) > 1:
        metrics.top1_minus_top2 = float(last[order[0]] - last[order[1]])


def _record_window_work(
    model,
    metrics: R2T2StepMetrics,
    *,
    mel_frames: int,
    audio_tokens: int,
    prompt_tokens: int,
    vocab_positions: int,
) -> None:
    """Count executed work for one window on the full-refeed path."""

    chunk_size = int(model.audio_tower.n_window) * 2
    chunk_lengths = [
        min(chunk_size, mel_frames - start)
        for start in range(0, mel_frames, chunk_size)
    ]
    block_count = 0
    if chunk_lengths:
        aftercnn = [
            int(_get_feat_extract_output_lengths(length))
            for length in chunk_lengths
        ]
        block_count = len(
            _attention_block_lengths(
                total_length=int(sum(aftercnn)),
                max_chunk_aftercnn=max(aftercnn),
                n_window=int(model.audio_tower.n_window),
                n_window_infer=int(model.audio_tower.n_window_infer),
            )
        )
    metrics.mel_frames = int(mel_frames)
    metrics.audio_tokens = int(audio_tokens)
    metrics.prompt_tokens = int(prompt_tokens)
    metrics.add_counter(MEL_FRAMES, int(mel_frames))
    metrics.add_counter(ENCODER_BLOCKS, block_count)
    metrics.add_counter(ENCODER_TOKENS, int(audio_tokens))
    metrics.add_counter(PREFILL_POSITIONS, int(prompt_tokens))
    metrics.add_counter(VOCAB_POSITIONS, int(vocab_positions))
    metrics.add_counter(CACHE_ALLOCATIONS)


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
    metrics: R2T2StepMetrics | None = None,
    decoder_compute: Qwen3ASRDecodeCompute | None = None,
    logits_observer: LogitsObserver | None = None,
) -> list[int]:
    """Decode only the completion; the prompt prefix is already in the cache."""

    eos = set(eos_token_ids)
    if logits_observer is not None:
        logits_observer(0, prefill.logits[0, -1, :])
    next_token = _token_id(greedy_next_token(prefill.logits))
    generated: list[int] = []
    for index in range(max_new_tokens):
        if next_token in eos:
            break
        generated.append(next_token)
        if index == max_new_tokens - 1:
            break
        with stage(metrics, "decode_step"):
            step = model.decode_step(
                input_ids=mx.array([[next_token]], dtype=mx.int32),
                kv_cache=prefill.past_key_values,
                compute=decoder_compute,
            )
            mx.eval(step.logits)
        if metrics is not None:
            metrics.add_counter(DECODE_STEPS)
        if logits_observer is not None:
            logits_observer(index + 1, step.logits[0, -1, :])
        next_token = _token_id(greedy_next_token(step.logits))
    return generated


def _token_id(token: mx.array) -> int:
    return int(np.array(token).reshape(-1)[0])


__all__ = [
    "Confucius4R2T2Adapter",
    "Confucius4R2T2StreamSession",
    "DEFAULT_STREAM_COMPUTE",
    "FULL_REFEED_STREAM_COMPUTE",
    "R2T2StreamCompute",
    "REFERENCE_STREAM_COMPUTE",
]
