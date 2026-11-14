"""Greedy local inference for Qwen3-ASR."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..models.qwen3_asr.config import Qwen3ASRConfig
from ..models.qwen3_asr.model import Qwen3ASRModel, replace_audio_embeddings
from ..models.qwen3_asr.processor import Qwen3ASRProcessor, Qwen3ASRPrompt, parse_asr_output
from ..models.qwen3_asr.text_decoder import NATIVE_DECODE_COMPUTE


TOKENIZER_EOS_FALLBACKS = ("<|im_end|>", "<|endoftext|>")


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
    eos_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "Qwen3ASRTranscriber":
        from ..models.qwen3_asr.checkpoint import (
            get_quantization_config,
            load_checkpoint_into_model,
            load_qwen3_asr_checkpoint,
            quantize_qwen3_asr_model,
        )

        model_dir = Path(model_dir)
        checkpoint = load_qwen3_asr_checkpoint(model_dir)
        if checkpoint.config.thinker_config.is_forced_aligner:
            raise ValueError(
                f"{model_dir} is a Qwen3-ForcedAligner checkpoint; it predicts time "
                "classes, not tokens. Use mlx_speech.align.load()."
            )
        model = Qwen3ASRModel(checkpoint.config)
        quantization = get_quantization_config(checkpoint.config)
        if quantization is not None:
            quantize_qwen3_asr_model(
                model, quantization, state_dict=checkpoint.state_dict
            )
        load_checkpoint_into_model(model, checkpoint, strict=True)
        model.set_dtype(dtype)
        model.eval()
        mx.eval(model.parameters())
        processor = Qwen3ASRProcessor.from_dir(model_dir)
        return cls(
            model=model,
            processor=processor,
            config=checkpoint.config,
            eos_token_ids=resolve_eos_token_ids(
                checkpoint.config,
                processor.tokenizer,
                model_dir=model_dir,
            ),
        )

    def transcribe(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        context: str = "",
        max_new_tokens: int = 448,
    ) -> Qwen3ASRResult:
        waveform, prompt = self._prepare_audio(
            audio, sample_rate=sample_rate, language=language,
            context=context, max_new_tokens=max_new_tokens,
        )
        return self._transcribe_group([waveform], [prompt], max_new_tokens=max_new_tokens)[0]

    def transcribe_batch(
        self,
        audios: Sequence[np.ndarray | mx.array | str | Path],
        *,
        sample_rate: int = 16000,
        language: str | None | Sequence[str | None] = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
    ) -> list[Qwen3ASRResult]:
        """Greedy transcription, grouped by sample count and returned in input order.

        Arrays must be 1-D mono at 16 kHz; ``sample_rate`` is checked, not
        resampled. Paths are loaded as mono and resampled to 16 kHz regardless
        of ``sample_rate``. Defaults: 48 clips/group, 448 tokens/clip. All inputs
        are validated before model inference.
        """
        if not isinstance(audios, (list, tuple)):
            raise TypeError("audios must be a list or tuple of waveforms or paths.")
        batch_size = 48 if batch_size is None else batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive int.")
        max_new_tokens = 448 if max_new_tokens is None else max_new_tokens
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive int.")
        if isinstance(language, str) or language is None:
            languages = [language] * len(audios)
        else:
            languages = list(language)
            if len(languages) != len(audios):
                raise ValueError("language sequence length must match len(audios).")

        # Preflight every row, including paths, before any audio/text model call.
        waveforms: list[np.ndarray] = []
        prompts: list[Qwen3ASRPrompt] = []
        for index, (audio, row_language) in enumerate(zip(audios, languages, strict=True)):
            try:
                waveform, prompt = self._prepare_audio(
                    audio, sample_rate=sample_rate, language=row_language,
                    context="", max_new_tokens=max_new_tokens,
                )
            except (TypeError, ValueError, OSError, RuntimeError) as exc:
                error_type = TypeError if isinstance(exc, TypeError) else ValueError
                raise error_type(f"Qwen3-ASR input {index}: {exc}") from exc
            waveforms.append(waveform)
            prompts.append(prompt)

        results: list[Qwen3ASRResult | None] = [None] * len(audios)
        order = sorted(range(len(audios)), key=lambda i: len(waveforms[i]))
        for start in range(0, len(order), batch_size):
            group = order[start : start + batch_size]
            outputs = self._transcribe_group(
                [waveforms[i] for i in group],
                [prompts[i] for i in group],
                max_new_tokens=max_new_tokens,
            )
            for index, result in zip(group, outputs, strict=True):
                results[index] = result
        return [result for result in results if result is not None]

    def _prepare_audio(
        self,
        audio: np.ndarray | mx.array | str | Path,
        *,
        sample_rate: int,
        language: str | None,
        context: str,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, Qwen3ASRPrompt]:
        if not isinstance(audio, (np.ndarray, mx.array, str, Path)):
            raise TypeError("expected a NumPy/MLX waveform array or an audio path.")
        if not isinstance(language, str) and language is not None:
            raise TypeError("language must be a string or None.")
        language = self.processor.resolve_language(language)
        if not isinstance(audio, (str, Path)) and np.iscomplexobj(audio):
            raise ValueError("waveform must contain real samples.")
        extractor = self.processor.feature_extractor
        waveform = extractor._audio_to_numpy(audio, sample_rate=sample_rate)
        if waveform.ndim != 1 or not waveform.size or not np.all(np.isfinite(waveform)):
            raise ValueError("expected a non-empty, finite 1-D mono waveform.")
        frames, audio_length = extractor.preflight_shape(len(waveform))
        if frames <= 0 or audio_length <= 0:
            raise ValueError("waveform is too short to produce audio features.")
        prompt = self.processor.build_prompt(
            context=context, audio_length=audio_length, language=language
        )
        validate_context_window(
            prompt_tokens=len(prompt.input_ids), max_new_tokens=max_new_tokens,
            max_position_embeddings=self.config.text_config.max_position_embeddings,
        )
        return waveform, prompt

    def _transcribe_group(
        self,
        waveforms: list[np.ndarray],
        prompts: list[Qwen3ASRPrompt],
        *,
        max_new_tokens: int,
    ) -> list[Qwen3ASRResult]:
        feature_rows = []
        for waveform in waveforms:
            # Keep each waveform's reflect-padded STFT boundary intact.
            features = self.processor.feature_extractor(
                waveform, sample_rate=self.processor.feature_extractor.sample_rate
            )
            feature_rows.append(mx.array(features.input_features[0]))
        audio_rows = self.model.get_audio_features_batch(feature_rows)
        for audio_features, prompt in zip(audio_rows, prompts, strict=True):
            if audio_features.shape[0] != prompt.audio_length:
                raise RuntimeError("Qwen3-ASR encoded audio length does not match prompt preflight.")

        lengths = [len(prompt.input_ids) for prompt in prompts]
        width = max(lengths)
        pads = mx.array([width - length for length in lengths], dtype=mx.int32)
        input_rows = [
            [0] * (width - length) + prompt.input_ids
            for prompt, length in zip(prompts, lengths, strict=True)
        ]
        input_ids = mx.array(input_rows, dtype=mx.int32)
        inputs_embeds = replace_audio_embeddings(
            input_ids, self.model.embed_input_ids(input_ids), audio_rows,
            audio_token_id=self.config.audio_token_id,
            audio_token_positions=[
                [index for index, token in enumerate(row) if token == self.config.audio_token_id]
                for row in input_rows
            ],
        )
        padded = min(lengths) != width
        validity = None
        positions = None
        if padded:
            slots = mx.arange(width)[None, :]
            prompt_mask = slots >= pads[:, None]
            positions = mx.maximum(slots - pads[:, None], 0)
            validity = mx.concatenate(
                [prompt_mask, mx.ones((len(prompts), max_new_tokens), dtype=mx.bool_)], axis=1
            )
        prefill = self.model.prefill(
            inputs_embeds=inputs_embeds,
            max_cache_len=width + min(64, max_new_tokens),
            attention_mask=validity[:, :width] if padded else None,
            position_ids=positions,
            last_logits_only=True,
            compute=NATIVE_DECODE_COMPUTE,
        )
        if prefill.past_key_values is None:
            raise RuntimeError("Qwen3-ASR prefill did not return a KV cache.")
        eos = set(self.eos_token_ids or resolve_eos_token_ids(self.config, self.processor.tokenizer))
        generated: list[list[int]] = [[] for _ in prompts]
        active = list(range(len(prompts)))
        finished = [False] * len(active)
        finished_count = 0
        live_mask = None
        dummy = mx.array(next(iter(eos), 0), dtype=mx.int32)
        token_vector = greedy_next_token(prefill.logits)
        mx.async_eval(token_vector)
        cache = prefill.past_key_values
        del prefill
        decode_positions = (
            mx.array(lengths, dtype=mx.int32)[:, None]
            + mx.arange(max_new_tokens, dtype=mx.int32)[None, :]
            if padded else None
        )
        for step in range(max_new_tokens):
            queued = None
            if step < max_new_tokens - 1:
                # Keep one prediction in flight before the host reads this
                # step's tokens. A just-finished row may do one wasted step.
                decode_ids = token_vector if live_mask is None else mx.where(
                    live_mask, token_vector, dummy
                )
                output = self.model.decode_step(
                    input_ids=decode_ids[:, None],
                    kv_cache=cache,
                    attention_mask=validity[:, : width + step + 1] if padded else None,
                    position_ids=decode_positions[:, step : step + 1] if padded else None,
                    compute=NATIVE_DECODE_COMPUTE,
                )
                queued = greedy_next_token(output.logits)
                mx.async_eval(queued)

            # Exactly one host read, including EOS bookkeeping for all rows.
            tokens = token_vector.tolist()
            newly_finished = []
            for row, token in enumerate(tokens):
                if not finished[row]:
                    if token in eos:
                        finished[row] = True
                        newly_finished.append(row)
                        finished_count += 1
                    else:
                        generated[active[row]].append(token)
            if finished_count == len(active) or queued is None:
                break

            # The queued step has already appended its KV position. Compact
            # it and its predicted tokens together, then grow only survivors.
            grow = cache.current_length == cache.max_length and step < max_new_tokens - 2
            if finished_count and (finished_count * 4 >= len(active) or grow):
                keep = [row for row, done in enumerate(finished) if not done]
                rows = mx.array(keep, dtype=mx.int32)
                cache.select_rows(rows)
                queued = queued[rows]
                if padded:
                    validity = validity[rows]
                    decode_positions = decode_positions[rows]
                active = [active[row] for row in keep]
                finished = [False] * len(active)
                finished_count = 0
                live_mask = None
            elif newly_finished:
                if live_mask is None:
                    live_mask = mx.ones((len(active),), dtype=mx.bool_)
                live_mask[mx.array(newly_finished, dtype=mx.int32)] = False
            if grow:
                cache.reserve(width + min(2 * (cache.max_length - width), max_new_tokens))
            token_vector = queued
        results = []
        for prompt, tokens in zip(prompts, generated, strict=True):
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            detected_language, text = parse_asr_output(raw, user_language=prompt.language)
            results.append(Qwen3ASRResult(
                text=text, tokens=tokens, language=detected_language, raw_text=raw,
                prompt_tokens=len(prompt.input_ids),
            ))
        return results

    def generate(self, audio: np.ndarray | mx.array | str | Path, **kwargs: Any) -> Qwen3ASRResult:
        return self.transcribe(audio, **kwargs)


def _eos_token_ids(eos_token_id: int | list[int] | None) -> set[int]:
    return set(_normalize_eos_token_ids(eos_token_id))


def resolve_eos_token_ids(
    config: Qwen3ASRConfig,
    tokenizer: Any,
    *,
    model_dir: str | Path | None = None,
) -> tuple[int, ...]:
    if model_dir is not None:
        ids = _generation_config_eos_token_ids(Path(model_dir))
        if ids:
            return ids

    ids = _normalize_eos_token_ids(config.text_config.eos_token_id)
    if ids:
        return ids

    return _tokenizer_eos_token_ids(tokenizer)


def _generation_config_eos_token_ids(model_dir: Path) -> tuple[int, ...]:
    config_path = model_dir / "generation_config.json"
    if not config_path.exists():
        return ()
    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return _normalize_eos_token_ids(payload.get("eos_token_id"))


def _normalize_eos_token_ids(eos_token_id: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, (list, tuple)):
        return tuple(dict.fromkeys(int(token_id) for token_id in eos_token_id))
    return (int(eos_token_id),)


def _tokenizer_eos_token_ids(tokenizer: Any) -> tuple[int, ...]:
    ids: list[int] = []
    for token in TOKENIZER_EOS_FALLBACKS:
        token_id = None
        if hasattr(tokenizer, "token_to_id"):
            token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            ids.append(int(token_id))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        ids.append(int(eos_token_id))
    return tuple(dict.fromkeys(ids))
