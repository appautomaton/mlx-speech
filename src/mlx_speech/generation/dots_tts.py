"""End-to-end pure-MLX generation for dots.tts SOAR and MeanFlow models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Literal

import mlx.core as mx
import numpy as np

from ..models.dots_tts.checkpoint import (
    LoadedDotsTTSComponents,
    load_dots_tts_components,
)
from ..models.dots_tts.dit_inference import (
    CachedDiTSolver,
    CachedMeanFlowSolver,
    CachedSOARSolver,
    DiTSolverState,
)
from ..models.dots_tts.semantic_encoder import SemanticEncoderState
from ..models.dots_tts.solvers import MeanFlowSolver, SOARSolver
from ..models.dots_tts.speaker import SpeakerFrontend
from ..models.dots_tts.text import (
    DotsTTSSchedule,
    DotsTTSTokenizer,
    build_generation_schedule,
    prepare_conditioned_text,
)


SAMPLE_RATE = 48_000
DEFAULT_MAX_AUDIO_PATCHES = 500
_MAX_AUDIO_PATCHES = 512
_RESAMPLE_LOWPASS_WIDTH = 64
_RESAMPLE_ROLLOFF = 0.95
_RESAMPLE_KAISER_BETA = 14.769656459379492
_RESAMPLE_WORKSPACE_BYTES = 32 * 1024 * 1024
_RESAMPLE_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _ResamplePlan:
    output_length: int
    width: int
    tap_count: int
    tile_size: int
    workspace_bytes: int


@dataclass(frozen=True)
class DotsTTSPromptConditioning:
    """Prepared reference state for one generation request."""

    speaker_condition: mx.array | None
    prompt_patches: mx.array | None
    prompt_latents: mx.array | None

    @property
    def prompt_patch_count(self) -> int:
        return 0 if self.prompt_patches is None else int(self.prompt_patches.shape[1])


@dataclass(frozen=True)
class DotsTTSSynthesisOutput:
    """Waveform and bounded generation metadata returned by dots.tts."""

    waveform: mx.array
    sample_rate: int
    num_patches: int


@dataclass(frozen=True)
class _DotsTTSStreamChunk:
    waveform: mx.array
    num_patches: int


@dataclass
class _DotsTTSRequestRNG:
    key: mx.array

    def next_key(self) -> mx.array:
        keys = mx.random.split(self.key)
        self.key = keys[0]
        return keys[1]


@dataclass
class _DotsTTSRequestState:
    """All mutable acoustic-generation state owned by one synthesis request."""

    fm_chunks: list[mx.array]
    cfg_chunks: list[mx.array]
    qwen_cache: Any
    semantic_state: SemanticEncoderState | None
    dit_solver: CachedDiTSolver | None
    dit_state: DiTSolverState | None
    rng: _DotsTTSRequestRNG | None
    generated_patches: int = 0

    def close(self) -> None:
        self.fm_chunks.clear()
        self.cfg_chunks.clear()
        self.qwen_cache = None
        self.semantic_state = None
        self.dit_solver = None
        self.dit_state = None
        self.rng = None


@dataclass
class _DotsTTSStreamState:
    """Mutable recurrent-vocoder state owned only by a streaming sink."""

    vocoder_state: Any
    vocoder_buffer: list[mx.array]
    pending_chunk_patches: int = 0
    yielded_waveform: bool = False
    has_non_silent_audio: bool = False


def _build_fm_attention_mask(fm_sequence_length: int, patch_size: int) -> mx.array:
    """Build the official causal-history/full-latent-block DiT mask."""

    if fm_sequence_length <= 0 or patch_size <= 0:
        raise ValueError("FM sequence length and patch size must be positive")
    total = fm_sequence_length + patch_size
    block_start = fm_sequence_length - 1
    rows = []
    if block_start > 0:
        causal = mx.arange(block_start)[:, None] >= mx.arange(block_start)[None, :]
        rows.append(
            mx.concatenate(
                (causal, mx.zeros((block_start, total - block_start), dtype=mx.bool_)),
                axis=1,
            )
        )
    rows.append(mx.ones((total - block_start, total), dtype=mx.bool_))
    mask = mx.concatenate(rows, axis=0)
    return mask[None]


def _build_fm_positions(fm_sequence_length: int, patch_size: int) -> mx.array:
    if fm_sequence_length <= 0 or patch_size <= 0:
        raise ValueError("FM sequence length and patch size must be positive")
    return mx.arange(fm_sequence_length + patch_size, dtype=mx.float32)[None]


def _concatenate_suffix(chunks: list[mx.array], length: int) -> mx.array:
    """Materialize exactly the newest DiT tokens without joining full history."""

    if length <= 0:
        raise ValueError("DiT suffix length must be positive")
    selected: list[mx.array] = []
    remaining = length
    for chunk in reversed(chunks):
        chunk_length = int(chunk.shape[1])
        if chunk_length <= 0:
            continue
        take = min(remaining, chunk_length)
        selected.append(chunk[:, chunk_length - take :])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(
            f"dots.tts DiT history has {length - remaining} tokens; needs {length}"
        )
    return mx.concatenate(tuple(reversed(selected)), axis=1)


def _sample_rate(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer sample rate")
    return int(value)


def _resample_plan(
    sample_count: int,
    source_rate: int,
    target_rate: int,
    *,
    max_output_samples: int | None = None,
) -> _ResamplePlan:
    source_rate = _sample_rate(source_rate, "source_rate")
    target_rate = _sample_rate(target_rate, "target_rate")
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise ValueError("sample_count must be a non-negative integer")
    sample_count = int(sample_count)
    if sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if max_output_samples is not None:
        if (
            isinstance(max_output_samples, bool)
            or not isinstance(max_output_samples, Integral)
            or max_output_samples < 0
        ):
            raise ValueError("max_output_samples must be a non-negative integer")
        max_output_samples = int(max_output_samples)

    output_length = (sample_count * target_rate + source_rate - 1) // source_rate
    if max_output_samples is not None:
        output_length = min(output_length, max_output_samples)
    if output_length * np.dtype(np.float32).itemsize > _RESAMPLE_MAX_OUTPUT_BYTES:
        raise ValueError(
            "dots.tts resampled output exceeds the bounded allocation limit: "
            f"samples={output_length}, bytes={output_length * 4}, "
            f"limit={_RESAMPLE_MAX_OUTPUT_BYTES}"
        )

    # rolloff is exactly 19/20, so this avoids float overflow in workspace planning.
    width_numerator = _RESAMPLE_LOWPASS_WIDTH * 20 * source_rate
    width_denominator = 19 * min(source_rate, target_rate)
    width = (width_numerator + width_denominator - 1) // width_denominator
    tap_count = 2 * width + 1
    if output_length == 0 or source_rate == target_rate:
        return _ResamplePlan(output_length, width, tap_count, 0, 0)
    if max(source_rate, target_rate) > np.iinfo(np.int64).max:
        raise ValueError(
            "dots.tts sample rates exceed the supported integer index range"
        )

    fixed_bytes = tap_count * np.dtype(np.int64).itemsize
    # Per row: quotient/remainder, tap indices, validity, float32 samples, and
    # float64 distance/Kaiser/sinc/reduction plus two ufunc scratch matrices.
    row_bytes = 2 * np.dtype(np.int64).itemsize + tap_count * (
        np.dtype(np.int64).itemsize
        + np.dtype(np.bool_).itemsize
        + np.dtype(np.float32).itemsize
        + 6 * np.dtype(np.float64).itemsize
    )
    available = _RESAMPLE_WORKSPACE_BYTES - fixed_bytes
    tile_size = min(output_length, available // row_bytes) if available >= 0 else 0
    if tile_size < 1:
        raise ValueError(
            "dots.tts resampling filter cannot fit one output row in the "
            f"{_RESAMPLE_WORKSPACE_BYTES}-byte workspace: taps={tap_count}"
        )
    workspace_bytes = fixed_bytes + tile_size * row_bytes
    return _ResamplePlan(
        output_length,
        width,
        tap_count,
        tile_size,
        workspace_bytes,
    )


def _source_prefix_length(
    sample_count: int,
    source_rate: int,
    target_rate: int,
    output_length: int,
    width: int,
) -> int:
    if output_length <= 0:
        return 0
    last_center = ((output_length - 1) * source_rate) // target_rate
    return min(sample_count, last_center + width + 1)


def _high_quality_resample(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
    *,
    max_output_samples: int | None = None,
) -> np.ndarray:
    """Match the official 64-wide Kaiser-windowed sinc reference resampler."""

    source_rate = _sample_rate(source_rate, "source_rate")
    target_rate = _sample_rate(target_rate, "target_rate")
    if not isinstance(waveform, np.ndarray):
        raise TypeError("dots.tts resampling expects a NumPy waveform")
    raw = waveform
    plan = _resample_plan(
        int(raw.size),
        source_rate,
        target_rate,
        max_output_samples=max_output_samples,
    )
    value = raw.astype(np.float32, copy=False).reshape(-1)
    if plan.output_length == 0:
        return value[:0]
    if source_rate == target_rate:
        return value[: plan.output_length].astype(np.float32, copy=False)

    output = np.empty(plan.output_length, dtype=np.float32)
    tap_offsets = np.arange(-plan.width, plan.width + 1, dtype=np.int64)
    scale = _RESAMPLE_ROLLOFF * min(1.0, target_rate / source_rate)
    kaiser_denominator = np.i0(_RESAMPLE_KAISER_BETA)
    sample_count = int(value.size)
    for start in range(0, plan.output_length, plan.tile_size):
        end = min(start + plan.tile_size, plan.output_length)
        count = end - start
        quotient = np.empty(count, dtype=np.int64)
        remainder = np.empty(count, dtype=np.int64)
        for offset in range(count):
            center, fraction = divmod(
                (start + offset) * source_rate,
                target_rate,
            )
            quotient[offset] = center
            remainder[offset] = fraction

        indices = quotient[:, None] + tap_offsets[None, :]
        valid = indices >= 0
        valid &= indices < sample_count
        np.clip(indices, 0, sample_count - 1, out=indices)
        samples = np.empty((count, plan.tap_count), dtype=np.float32)
        np.take(value, indices, out=samples)
        np.multiply(samples, valid, out=samples)

        distance = (
            tap_offsets[None, :] - (remainder.astype(np.float64) / target_rate)[:, None]
        )
        distance *= scale
        np.clip(
            distance,
            -_RESAMPLE_LOWPASS_WIDTH,
            _RESAMPLE_LOWPASS_WIDTH,
            out=distance,
        )
        kaiser = distance / _RESAMPLE_LOWPASS_WIDTH
        np.square(kaiser, out=kaiser)
        np.subtract(1.0, kaiser, out=kaiser)
        np.clip(kaiser, 0.0, None, out=kaiser)
        np.sqrt(kaiser, out=kaiser)
        kaiser *= _RESAMPLE_KAISER_BETA
        kaiser = np.i0(kaiser)
        kaiser /= kaiser_denominator
        sinc = np.sinc(distance)
        coefficients = (sinc * kaiser * scale).astype(np.float32)
        np.multiply(samples, coefficients, out=samples)
        output[start:end] = np.sum(samples, axis=1, dtype=np.float32)
    return output


def _mono_waveform(value: mx.array) -> mx.array:
    """Normalize accepted in-memory reference layouts to one mono channel."""

    if value.ndim == 1:
        waveform = value
    elif value.ndim == 2 and int(value.shape[0]) == 1:
        waveform = value[0]
    elif value.ndim == 2 and int(value.shape[0]) <= 8 < int(value.shape[1]):
        waveform = mx.mean(value.astype(mx.float32), axis=0)
    elif value.ndim == 2 and int(value.shape[1]) <= 8:
        waveform = mx.mean(value.astype(mx.float32), axis=1)
    elif value.ndim == 3 and tuple(int(size) for size in value.shape[:2]) == (1, 1):
        waveform = value[0, 0]
    else:
        raise ValueError(
            "dots.tts reference_audio must be mono samples, samples-by-channels, "
            f"or shape (1, 1, samples); got {value.shape}"
        )
    if int(waveform.size) == 0:
        raise ValueError("dots.tts reference_audio must not be empty")
    waveform = waveform.astype(mx.float32)
    mx.eval(waveform)
    if not bool(mx.all(mx.isfinite(waveform)).item()):
        raise ValueError("dots.tts reference_audio contains non-finite values")
    return waveform


class DotsTTSGenerator:
    """Compose converted dots.tts components into non-streaming waveform inference."""

    def __init__(
        self,
        components: LoadedDotsTTSComponents,
        tokenizer: DotsTTSTokenizer,
    ):
        self.components = components
        self.tokenizer = tokenizer
        self.config = components.layout.config
        self.sample_rate = self.config.vocoder.sample_rate
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"dots.tts generation requires {SAMPLE_RATE} Hz artifacts")
        self.speaker_frontend = SpeakerFrontend(
            max_audio_seconds=self.config.xvec_max_audio_seconds
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "DotsTTSGenerator":
        components = load_dots_tts_components(model_dir)
        tokenizer = DotsTTSTokenizer.from_dir(components.layout.model_dir)
        return cls(components, tokenizer)

    @property
    def _activation_dtype(self) -> mx.Dtype:
        return self.components.core.hidden_projection.weight.dtype

    def _load_reference(
        self,
        reference_audio: str | Path | mx.array,
        *,
        reference_sample_rate: int | None,
        prompt_patch_budget: int | None = None,
        max_output_samples: int | None = None,
    ) -> mx.array:
        if isinstance(reference_audio, (str, Path)):
            from ..audio import load_audio

            waveform, loaded_sample_rate = load_audio(
                reference_audio,
                mono=True,
            )
            source_rate = loaded_sample_rate
        else:
            waveform = reference_audio
            source_rate = (
                self.sample_rate
                if reference_sample_rate is None
                else reference_sample_rate
            )
        source_rate = _sample_rate(source_rate, "reference_sample_rate")
        mono = _mono_waveform(waveform)
        sample_count = int(mono.size)
        target_sample_count = (
            sample_count * self.sample_rate + source_rate - 1
        ) // source_rate
        if prompt_patch_budget is not None:
            prompt_patch_count = self._estimate_prompt_patch_count(target_sample_count)
            if prompt_patch_count <= 0:
                raise ValueError(
                    "dots.tts reference audio is too short for prompt prefill"
                )
            minimum_budget = prompt_patch_count + 2
            if prompt_patch_budget < minimum_budget:
                raise ValueError(
                    "dots.tts patch budget cannot fit prompt prefill, regenerated "
                    "prompt tail, and one payload patch: "
                    f"prompt_patches={prompt_patch_count}, "
                    f"budget={prompt_patch_budget}, minimum={minimum_budget}"
                )

        limited_plan = _resample_plan(
            sample_count,
            source_rate,
            self.sample_rate,
            max_output_samples=max_output_samples,
        )
        if limited_plan.output_length < target_sample_count:
            source_prefix = _source_prefix_length(
                sample_count,
                source_rate,
                self.sample_rate,
                limited_plan.output_length,
                limited_plan.width,
            )
            mono = mono[:source_prefix]
        resampled = _high_quality_resample(
            np.asarray(mono, dtype=np.float32),
            source_rate,
            self.sample_rate,
            max_output_samples=max_output_samples,
        )
        return mx.array(resampled, dtype=mx.float32)

    def _estimate_prompt_patch_count(self, sample_count: int) -> int:
        if sample_count <= 0:
            raise ValueError("dots.tts reference_audio must not be empty")
        samples_per_patch = self.config.patch_size * self.config.vocoder.hop_size
        encoded_patch_count = math.ceil(sample_count / samples_per_patch)
        return encoded_patch_count - 1

    def _speaker_condition(
        self,
        waveform: mx.array,
        *,
        speaker_scale: float,
    ) -> mx.array:
        if not math.isfinite(speaker_scale):
            raise ValueError("speaker_scale must be finite")
        features, length = self.speaker_frontend.features(
            np.asarray(waveform, dtype=np.float32),
            sample_rate=self.sample_rate,
        )
        embedding = self.components.speaker_encoder(
            mx.array(features[None], dtype=mx.float32),
            lengths=mx.array([length], dtype=mx.int32),
        )
        core = self.components.core
        condition = core.speaker_projection_norm(
            core.speaker_projection(embedding * float(speaker_scale))
        )
        return condition.astype(self._activation_dtype)

    def _prompt_latents(
        self,
        waveform: mx.array,
        *,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        samples_per_patch = self.config.patch_size * self.config.vocoder.hop_size
        sample_count = int(waveform.shape[0])
        padded_count = math.ceil(sample_count / samples_per_patch) * samples_per_patch
        if padded_count > sample_count:
            waveform = mx.pad(waveform, (0, padded_count - sample_count))
        distribution = self.components.audio_vae.encode(waveform[None, None])
        noise_shape = (
            int(distribution.shape[0]),
            self.config.latent_dim,
            int(distribution.shape[2]),
        )
        sampled = self.components.latent_io.sample_distribution(
            distribution,
            noise=mx.random.normal(noise_shape, key=key),
        )
        if int(sampled.shape[1]) <= self.config.patch_size:
            raise ValueError("dots.tts reference audio is too short for prompt prefill")
        sampled = sampled[:, : -self.config.patch_size]
        usable = (
            int(sampled.shape[1]) // self.config.patch_size
        ) * self.config.patch_size
        if usable <= 0:
            raise ValueError(
                "dots.tts reference audio produced no complete prompt patch"
            )
        prompt_latents = sampled[:, :usable]
        normalized = self.components.latent_io.normalize(prompt_latents)
        prompt_patches = normalized.reshape(
            1,
            usable // self.config.patch_size,
            self.config.patch_size,
            self.config.latent_dim,
        )
        return (
            prompt_patches.astype(self._activation_dtype),
            prompt_latents.astype(self._activation_dtype),
        )

    def prepare_prompt(
        self,
        reference_audio: str | Path | mx.array | None,
        *,
        reference_text: str | None,
        reference_sample_rate: int | None = None,
        speaker_scale: float = 1.5,
        max_audio_patches: int | None = None,
        _rng: _DotsTTSRequestRNG | None = None,
    ) -> DotsTTSPromptConditioning:
        """Prepare continuation, speaker-only, or no-reference conditioning.

        A non-empty reference transcript enables in-context latent prefill. Reference
        audio without a transcript deliberately uses only the speaker embedding.
        """

        if reference_audio is None:
            if reference_text is not None and reference_text.strip():
                raise ValueError("dots.tts reference_text requires reference_audio")
            return DotsTTSPromptConditioning(None, None, None)
        has_reference_text = reference_text is not None and bool(reference_text.strip())
        speaker_sample_limit = round(
            self.sample_rate * self.speaker_frontend.max_audio_seconds
        )
        waveform = self._load_reference(
            reference_audio,
            reference_sample_rate=reference_sample_rate,
            prompt_patch_budget=max_audio_patches if has_reference_text else None,
            max_output_samples=None if has_reference_text else speaker_sample_limit,
        )
        prompt_patch_count = 0
        if has_reference_text:
            prompt_patch_count = self._estimate_prompt_patch_count(
                int(waveform.shape[0])
            )
            if prompt_patch_count <= 0:
                raise ValueError(
                    "dots.tts reference audio is too short for prompt prefill"
                )
            minimum_budget = prompt_patch_count + 2
            if max_audio_patches is not None and max_audio_patches < minimum_budget:
                raise ValueError(
                    "dots.tts patch budget cannot fit prompt prefill, regenerated "
                    "prompt tail, and one payload patch: "
                    f"prompt_patches={prompt_patch_count}, "
                    f"budget={max_audio_patches}, minimum={minimum_budget}"
                )
        speaker_condition = self._speaker_condition(
            waveform,
            speaker_scale=speaker_scale,
        )
        if not has_reference_text:
            return DotsTTSPromptConditioning(speaker_condition, None, None)
        rng = _DotsTTSRequestRNG(mx.random.key(0)) if _rng is None else _rng
        prompt_patches, prompt_latents = self._prompt_latents(
            waveform,
            key=rng.next_key(),
        )
        if int(prompt_patches.shape[1]) != prompt_patch_count:
            raise RuntimeError(
                "dots.tts prompt patch estimate differs from AudioVAE output: "
                f"estimated={prompt_patch_count}, actual={prompt_patches.shape[1]}"
            )
        return DotsTTSPromptConditioning(
            speaker_condition,
            prompt_patches,
            prompt_latents,
        )

    def _append_hidden(
        self,
        hidden: mx.array,
        fm_chunks: list[mx.array],
        cfg_chunks: list[mx.array],
    ) -> None:
        core = self.components.core
        last = hidden[:, -1:, :]
        fm_chunks.append(core.hidden_projection(last).astype(self._activation_dtype))
        cfg_chunks.append(
            core.hidden_projection(mx.zeros_like(last)).astype(self._activation_dtype)
        )

    def _append_history(
        self,
        patch: mx.array,
        fm_chunks: list[mx.array],
        cfg_chunks: list[mx.array],
    ) -> None:
        projected = self.components.core.latent_projection(patch).astype(
            self._activation_dtype
        )
        fm_chunks.append(projected)
        cfg_chunks.append(projected)

    def _prefill(
        self,
        schedule: DotsTTSSchedule,
        prompt: DotsTTSPromptConditioning,
        state: _DotsTTSRequestState,
    ) -> tuple[int, mx.array]:
        prompt_count = prompt.prompt_patch_count
        if schedule.audio_patch_budget <= prompt_count:
            raise ValueError(
                "dots.tts patch budget must include prompt spans and one decode span"
            )
        prefill_end = schedule.audio_span_positions[prompt_count]
        schedule_array = mx.array(schedule.token_ids, dtype=mx.int32)[None]
        input_ids = schedule_array[:, :prefill_end]
        inputs_embeds = self.components.core.qwen.get_input_embeddings()(input_ids)
        if prompt_count:
            patch_embeddings, state.semantic_state = (
                self.components.core.semantic_encoder.prefill(
                    prompt.prompt_latents,
                    max_audio_patches=schedule.audio_patch_budget,
                )
            )
            positions = mx.array(
                schedule.audio_span_positions[:prompt_count], dtype=mx.int32
            )
            inputs_embeds[:, positions, :] = patch_embeddings[:, :prompt_count].astype(
                inputs_embeds.dtype
            )
        qwen_output = self.components.core.qwen.step(
            inputs_embeds=inputs_embeds,
            cache=None,
            cache_capacity=len(schedule.token_ids),
        )
        state.qwen_cache = qwen_output.cache
        hidden = qwen_output.last_hidden_state
        cursor = 0
        for prompt_index, span_position in enumerate(
            schedule.audio_span_positions[:prompt_count]
        ):
            if span_position > cursor:
                self._append_hidden(
                    hidden[:, span_position - 1 : span_position],
                    state.fm_chunks,
                    state.cfg_chunks,
                )
            self._append_history(
                prompt.prompt_patches[:, prompt_index],
                state.fm_chunks,
                state.cfg_chunks,
            )
            if span_position + 1 < len(schedule.token_ids) and schedule.token_ids[
                span_position + 1
            ] in {self.tokenizer.audio_gen_span_id, self.tokenizer.audio_comp_span_id}:
                self._append_hidden(
                    hidden[:, span_position : span_position + 1],
                    state.fm_chunks,
                    state.cfg_chunks,
                )
            cursor = span_position + 1
        if prefill_end > cursor:
            self._append_hidden(
                hidden[:, prefill_end - 1 : prefill_end],
                state.fm_chunks,
                state.cfg_chunks,
            )
        return prefill_end, hidden[:, -1:]

    def _new_dit_request(
        self,
        max_audio_patches: int,
    ) -> tuple[CachedDiTSolver | None, DiTSolverState | None]:
        dit = self.components.core.dit
        # Explicit structural test doubles may opt into the full-history oracle.
        # Production components never set this marker, so cached-solver errors
        # propagate instead of silently changing inference paths.
        if getattr(dit, "_dots_tts_full_history_test_double", False):
            return None, None
        solver_type = (
            CachedMeanFlowSolver if self.config.mode == "meanflow" else CachedSOARSolver
        )
        solver = solver_type(
            dit,
            self.components.core.coordinate_projection,
            latent_dim=self.config.latent_dim,
            patch_size=self.config.patch_size,
        )
        return solver, solver.new_state(max_audio_patches)

    def _solve_patch(
        self,
        state: _DotsTTSRequestState,
        *,
        speaker_condition: mx.array | None,
        solver_steps: int | None,
        guidance_scale: float,
    ) -> mx.array:
        fm_sequence_length = sum(int(chunk.shape[1]) for chunk in state.fm_chunks)
        padding = mx.zeros(
            (1, self.config.patch_size, self.config.dit.hidden_size),
            dtype=self._activation_dtype,
        )
        if state.rng is None:
            raise RuntimeError("dots.tts request RNG was released before completion")
        noise = mx.random.normal(
            (1, self.config.patch_size, self.config.latent_dim),
            key=state.rng.next_key(),
        ).astype(self._activation_dtype)
        if state.dit_solver is not None:
            if state.dit_state is None:
                raise RuntimeError("dots.tts request is missing its DiT solver state")
            fresh_length = (
                state.dit_solver.unit_length + state.dit_solver.hidden_patch_size
            )
            cache = state.dit_state.cache
            if fm_sequence_length == fresh_length or cache is not None:
                fresh = _concatenate_suffix(state.fm_chunks, fresh_length)
                cfg_fresh = (
                    None
                    if self.config.mode == "meanflow"
                    else _concatenate_suffix(state.cfg_chunks, fresh_length)
                )
                return state.dit_solver.sample_tail(
                    state.dit_state,
                    previous_unit=fresh[:, : state.dit_solver.unit_length],
                    current_hidden=fresh[:, state.dit_solver.unit_length :],
                    cfg_previous_unit=(
                        None
                        if cfg_fresh is None
                        else cfg_fresh[:, : state.dit_solver.unit_length]
                    ),
                    cfg_current_hidden=(
                        None
                        if cfg_fresh is None
                        else cfg_fresh[:, state.dit_solver.unit_length :]
                    ),
                    speaker_condition=speaker_condition,
                    guidance_scale=guidance_scale,
                    steps=solver_steps,
                    noise=noise,
                )
            fm_sequence = mx.concatenate(state.fm_chunks, axis=1)
            sequence = mx.concatenate((fm_sequence, padding), axis=1)
            cfg_sequence = None
            if self.config.mode != "meanflow":
                cfg_sequence = mx.concatenate(
                    (mx.concatenate(state.cfg_chunks, axis=1), padding), axis=1
                )
            return state.dit_solver.sample(
                state.dit_state,
                sequence=sequence,
                cfg_sequence=cfg_sequence,
                attention_mask=None,
                positions=None,
                speaker_condition=speaker_condition,
                guidance_scale=guidance_scale,
                steps=solver_steps,
                noise=noise,
            )
        fm_sequence = mx.concatenate(state.fm_chunks, axis=1)
        sequence = mx.concatenate((fm_sequence, padding), axis=1)
        mask = _build_fm_attention_mask(fm_sequence_length, self.config.patch_size)
        positions = _build_fm_positions(fm_sequence_length, self.config.patch_size)
        if self.config.mode == "meanflow":
            solver = MeanFlowSolver(
                self.components.core.dit,
                self.components.core.coordinate_projection,
                latent_dim=self.config.latent_dim,
            )
            return solver.sample(
                sequence=sequence,
                attention_mask=mask,
                positions=positions,
                speaker_condition=speaker_condition,
                steps=solver_steps,
                patch_size=self.config.patch_size,
                noise=noise,
            )
        cfg_sequence = mx.concatenate(
            (mx.concatenate(state.cfg_chunks, axis=1), padding), axis=1
        )
        solver = SOARSolver(
            self.components.core.dit,
            self.components.core.coordinate_projection,
            latent_dim=self.config.latent_dim,
        )
        return solver.sample(
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            attention_mask=mask,
            positions=positions,
            speaker_condition=speaker_condition,
            guidance_scale=guidance_scale,
            steps=solver_steps,
            patch_size=self.config.patch_size,
            noise=noise,
        )

    def _decode_stream_chunk(
        self,
        state: _DotsTTSStreamState,
        patches: list[mx.array],
        *,
        final: bool = False,
    ) -> _DotsTTSStreamChunk | None:
        patch_count = len(patches)
        if patches:
            latent = mx.concatenate(patches, axis=1).transpose(0, 2, 1)
        else:
            latent = mx.zeros(
                (1, self.config.latent_dim, 0), dtype=self._activation_dtype
            )
        decoded, state.vocoder_state = self.components.audio_vae.decode_chunk(
            latent,
            state.vocoder_state,
            final=final,
        )
        if decoded.ndim != 3 or tuple(int(size) for size in decoded.shape[:2]) != (
            1,
            1,
        ):
            raise RuntimeError(
                f"dots.tts AudioVAE returned invalid waveform {decoded.shape}"
            )
        state.pending_chunk_patches += patch_count
        waveform = decoded[0, 0].astype(mx.float32)
        mx.eval(waveform)
        if not bool(mx.all(mx.isfinite(waveform)).item()):
            raise RuntimeError("dots.tts generation produced non-finite audio")
        if int(waveform.size) == 0:
            return None
        num_patches = state.pending_chunk_patches
        state.pending_chunk_patches = 0
        state.yielded_waveform = True
        if bool(mx.any(mx.abs(waveform) > 0).item()):
            state.has_non_silent_audio = True
        return _DotsTTSStreamChunk(
            waveform=waveform,
            num_patches=num_patches,
        )

    @staticmethod
    def _validate_max_audio_patches(max_audio_patches: int) -> int:
        if (
            isinstance(max_audio_patches, bool)
            or not isinstance(max_audio_patches, Integral)
            or max_audio_patches <= 0
        ):
            raise ValueError("max_audio_patches must be positive")
        max_audio_patches = int(max_audio_patches)
        if max_audio_patches > _MAX_AUDIO_PATCHES:
            raise ValueError(f"max_audio_patches must not exceed {_MAX_AUDIO_PATCHES}")
        return max_audio_patches

    @staticmethod
    def _validate_stream_chunk_patches(stream_chunk_patches: int) -> int:
        if (
            isinstance(stream_chunk_patches, bool)
            or not isinstance(stream_chunk_patches, Integral)
            or stream_chunk_patches <= 0
        ):
            raise ValueError("stream_chunk_patches must be a positive integer")
        return int(stream_chunk_patches)

    def _generate_latent_patches(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        max_audio_patches: int = DEFAULT_MAX_AUDIO_PATCHES,
        solver_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        eos_threshold: float = 0.8,
        template: Literal["tts", "tts_interleave"] = "tts",
    ) -> Iterator[mx.array]:
        """Yield payload latent patches from one request-local acoustic core.

        Transcript-backed references consume leading schedule spans for continuation
        prefill. Audio-only references leave the target-only schedule unchanged and
        apply speaker conditioning alone. No-reference generation keeps the official
        target-only parity path, but its random voice is not a quality-supported mode
        for the released multi-speaker checkpoints.
        """

        max_audio_patches = self._validate_max_audio_patches(max_audio_patches)
        if not 0.0 <= eos_threshold <= 1.0:
            raise ValueError("eos_threshold must be in [0, 1]")
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        rng = _DotsTTSRequestRNG(mx.random.key(int(seed)))
        prompt = self.prepare_prompt(
            reference_audio,
            reference_text=reference_text,
            reference_sample_rate=reference_sample_rate,
            speaker_scale=speaker_scale,
            max_audio_patches=max_audio_patches,
            _rng=rng,
        )
        prompt_text = reference_text if prompt.prompt_patch_count else None
        conditioned_prompt, conditioned_target = prepare_conditioned_text(
            text,
            language=language,
            prompt_text=prompt_text,
        )
        schedule = build_generation_schedule(
            text=f"{conditioned_prompt}{conditioned_target}",
            tokenizer=self.tokenizer,
            max_audio_patches=max_audio_patches,
            template=template,
        )
        max_positions = self.components.layout.qwen_config.max_position_embeddings
        if len(schedule.token_ids) > max_positions:
            raise ValueError(
                "dots.tts generation schedule exceeds Qwen max_position_embeddings: "
                f"schedule_length={len(schedule.token_ids)}, "
                f"max_position_embeddings={max_positions}"
            )
        minimum_budget = prompt.prompt_patch_count + (
            2 if prompt.prompt_patch_count else 1
        )
        if schedule.audio_patch_budget < minimum_budget:
            raise ValueError(
                "dots.tts patch budget leaves no payload after continuation prefill: "
                f"budget={schedule.audio_patch_budget}, minimum={minimum_budget}"
            )

        dit_solver, dit_state = self._new_dit_request(max_audio_patches)
        state = _DotsTTSRequestState(
            fm_chunks=[],
            cfg_chunks=[],
            qwen_cache=None,
            semantic_state=None,
            dit_solver=dit_solver,
            dit_state=dit_state,
            rng=rng,
        )
        try:
            position, hidden = self._prefill(
                schedule,
                prompt,
                state,
            )
            audio_ids = {
                self.tokenizer.audio_gen_span_id,
                self.tokenizer.audio_comp_span_id,
            }
            discard_regenerated_prompt_tail = prompt.prompt_patch_count > 0
            schedule_ids = schedule.token_ids
            while position < len(schedule_ids):
                if schedule_ids[position] not in audio_ids:
                    next_audio = next(
                        (
                            index
                            for index in range(position, len(schedule_ids))
                            if schedule_ids[index] in audio_ids
                        ),
                        len(schedule_ids),
                    )
                    output = self.components.core.qwen.step(
                        input_ids=mx.array(
                            schedule_ids[position:next_audio], dtype=mx.int32
                        )[None],
                        cache=state.qwen_cache,
                        cache_capacity=len(schedule_ids),
                    )
                    hidden, state.qwen_cache = output.last_hidden_state, output.cache
                    self._append_hidden(hidden, state.fm_chunks, state.cfg_chunks)
                    position = next_audio
                    continue

                stop_after = False
                if not discard_regenerated_prompt_tail:
                    stop_after = bool(
                        self.components.core.qwen.should_stop(
                            hidden[:, -1:], threshold=eos_threshold
                        ).item()
                    )
                patch = self._solve_patch(
                    state,
                    speaker_condition=prompt.speaker_condition,
                    solver_steps=solver_steps,
                    guidance_scale=guidance_scale,
                )
                mx.eval(patch)
                self._append_history(patch, state.fm_chunks, state.cfg_chunks)
                denormalized = self.components.latent_io.denormalize(patch)
                if state.semantic_state is None:
                    patch_embedding, state.semantic_state = (
                        self.components.core.semantic_encoder.prefill(
                            denormalized,
                            max_audio_patches=schedule.audio_patch_budget,
                        )
                    )
                else:
                    patch_embedding, state.semantic_state = (
                        self.components.core.semantic_encoder.decode_patch(
                            denormalized,
                            state.semantic_state,
                        )
                    )
                output = self.components.core.qwen.step(
                    inputs_embeds=patch_embedding.astype(self._activation_dtype),
                    cache=state.qwen_cache,
                    cache_capacity=len(schedule_ids),
                )
                hidden, state.qwen_cache = output.last_hidden_state, output.cache
                if (
                    position + 1 < len(schedule_ids)
                    and schedule_ids[position + 1] in audio_ids
                ):
                    self._append_hidden(hidden, state.fm_chunks, state.cfg_chunks)
                position += 1
                if discard_regenerated_prompt_tail:
                    discard_regenerated_prompt_tail = False
                else:
                    state.generated_patches += 1
                    yield denormalized
                if stop_after:
                    break

            if not state.generated_patches:
                raise RuntimeError(
                    "dots.tts generation produced no payload latent patches"
                )
        finally:
            state.close()

    def synthesize_stream(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        max_audio_patches: int = DEFAULT_MAX_AUDIO_PATCHES,
        solver_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        eos_threshold: float = 0.8,
        template: Literal["tts", "tts_interleave"] = "tts",
        stream_chunk_patches: int = 4,
    ) -> Iterator[_DotsTTSStreamChunk]:
        """Yield waveform chunks for one text segment without sentence splitting."""

        stream_chunk_patches = self._validate_stream_chunk_patches(stream_chunk_patches)
        max_audio_patches = self._validate_max_audio_patches(max_audio_patches)
        maximum_vocoder_chunk = (
            min(stream_chunk_patches, max_audio_patches) * self.config.patch_size
        )
        state = _DotsTTSStreamState(
            vocoder_state=self.components.audio_vae.init_decode_state(
                maximum_chunk_size=maximum_vocoder_chunk
            ),
            vocoder_buffer=[],
        )
        patches = self._generate_latent_patches(
            text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            reference_sample_rate=reference_sample_rate,
            max_audio_patches=max_audio_patches,
            solver_steps=solver_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            language=language,
            seed=seed,
            eos_threshold=eos_threshold,
            template=template,
        )
        try:
            generated_patches = 0
            for patch in patches:
                generated_patches += 1
                if generated_patches <= 2:
                    chunk = self._decode_stream_chunk(state, [patch])
                    if chunk is not None:
                        yield chunk
                else:
                    state.vocoder_buffer.append(patch)
                    if len(state.vocoder_buffer) == stream_chunk_patches:
                        chunk = self._decode_stream_chunk(
                            state,
                            state.vocoder_buffer,
                        )
                        state.vocoder_buffer = []
                        if chunk is not None:
                            yield chunk
            if state.vocoder_buffer:
                chunk = self._decode_stream_chunk(state, state.vocoder_buffer)
                state.vocoder_buffer = []
                if chunk is not None:
                    yield chunk
            tail = self._decode_stream_chunk(state, [], final=True)
            if tail is not None:
                yield tail
            if state.pending_chunk_patches:
                raise RuntimeError(
                    "dots.tts vocoder produced no audio for payload patches"
                )
            if not state.yielded_waveform:
                raise RuntimeError("dots.tts generation produced empty audio")
            if not state.has_non_silent_audio:
                raise RuntimeError("dots.tts generation produced silent audio")
        finally:
            patches.close()
            state.vocoder_buffer.clear()
            state.vocoder_state = None

    def synthesize(
        self,
        text: str,
        *,
        reference_audio: str | Path | mx.array | None = None,
        reference_text: str | None = None,
        reference_sample_rate: int | None = None,
        max_audio_patches: int = DEFAULT_MAX_AUDIO_PATCHES,
        solver_steps: int | None = None,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        language: str | None = None,
        seed: int = 42,
        eos_threshold: float = 0.8,
        template: Literal["tts", "tts_interleave"] = "tts",
        stream_chunk_patches: int = 4,
    ) -> DotsTTSSynthesisOutput:
        """Generate all payload latents and decode one complete waveform."""

        self._validate_stream_chunk_patches(stream_chunk_patches)
        patches = list(
            self._generate_latent_patches(
                text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sample_rate=reference_sample_rate,
                max_audio_patches=max_audio_patches,
                solver_steps=solver_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                language=language,
                seed=seed,
                eos_threshold=eos_threshold,
                template=template,
            )
        )
        latent = mx.concatenate(patches, axis=1).transpose(0, 2, 1)
        decoded = self.components.audio_vae.decode(latent)
        if decoded.ndim != 3 or tuple(int(size) for size in decoded.shape[:2]) != (
            1,
            1,
        ):
            raise RuntimeError(
                f"dots.tts AudioVAE returned invalid waveform {decoded.shape}"
            )
        waveform = decoded[0, 0].astype(mx.float32)
        mx.eval(waveform)
        if not bool(mx.all(mx.isfinite(waveform)).item()):
            raise RuntimeError("dots.tts generation produced non-finite audio")
        if int(waveform.size) == 0:
            raise RuntimeError("dots.tts generation produced empty audio")
        if not bool(mx.any(mx.abs(waveform) > 0).item()):
            raise RuntimeError("dots.tts generation produced silent audio")
        return DotsTTSSynthesisOutput(
            waveform=waveform,
            sample_rate=self.sample_rate,
            num_patches=len(patches),
        )


__all__ = [
    "DEFAULT_MAX_AUDIO_PATCHES",
    "DotsTTSGenerator",
    "DotsTTSPromptConditioning",
    "DotsTTSSynthesisOutput",
    "SAMPLE_RATE",
]
