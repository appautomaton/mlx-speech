"""Clock-synchronous live streaming for Audio8-ASR-Infinite."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from .cache import rebase_audio_cache, trim_text_cache
from .text_decoder import Audio8TextCache, text_cache_length

if TYPE_CHECKING:
    from .audio_encoder import Audio8AudioCache
    from .model import Audio8ASRModel


@dataclass(frozen=True)
class Audio8Result:
    text: str
    tokens: tuple[int, ...]
    language: str
    generated_token_ids: tuple[int, ...]
    semantic_vad_probabilities: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _AudioWindow:
    audio: np.ndarray
    token_count: int
    clock_end: int


class Audio8StreamSession:
    def __init__(
        self,
        model: Audio8ASRModel,
        *,
        language: str,
        frame_len: int,
        target_delay_ms: int,
        right_pad_text_tokens: int,
        rolling_context_tokens: int,
        rolling_trim_interval_ms: int,
    ) -> None:
        self.model = model
        self.language = language.strip().lower()
        self.frame_len = frame_len
        self.target_delay_ms = target_delay_ms
        self.num_delay_tokens = model.config.delay_tokens(frame_len, target_delay_ms)
        self.clock_ms = model.config.clock_ms(frame_len)
        self.samples_per_token = 16 * self.clock_ms
        self.left_pad_tokens = model.config.left_pad_tokens(frame_len)
        self.prefill_tokens = self.left_pad_tokens + self.num_delay_tokens + 1
        if right_pad_text_tokens < 0:
            raise ValueError("right_pad_text_tokens must be non-negative")
        self.right_pad_tokens = self.num_delay_tokens + 1 + int(right_pad_text_tokens)
        self.look_back_samples = 840
        self.look_ahead_samples = 40
        self.rolling_context_tokens = int(rolling_context_tokens)
        if self.rolling_context_tokens < 0:
            raise ValueError("rolling_context_tokens must be non-negative")
        self.stable_prefix_tokens = min(16, max(0, self.rolling_context_tokens - 1))
        self.rolling_trim_interval_tokens = max(
            1, (rolling_trim_interval_ms + self.clock_ms - 1) // self.clock_ms
        )
        self.rolling_trim_target = (
            self.rolling_context_tokens - self.rolling_trim_interval_tokens + 1
        )
        if (
            self.rolling_context_tokens
            and self.rolling_trim_target <= self.stable_prefix_tokens
        ):
            raise ValueError("Audio8 rolling window is too small for its stable prefix")
        self.audio_cache: Audio8AudioCache = model.audio_tower.new_cache(
            max_query_frames=frame_len
        )
        # The ring would drop the head of an oversized first prefill before
        # attention reads it.
        window_frames = self.audio_cache.layers[0].capacity
        if self.prefill_tokens * frame_len > window_frames:
            max_delay_ms = (
                window_frames // frame_len - self.left_pad_tokens - 1
            ) * self.clock_ms
            raise ValueError(
                f"target_delay_ms={target_delay_ms} overflows the {window_frames}-frame "
                f"audio attention window; the maximum for frame_len={frame_len} "
                f"is {max_delay_ms}"
            )
        self.text_cache: Audio8TextCache | None = None
        self.t_cond = model.build_t_cond(
            frame_len=frame_len,
            num_delay_tokens=self.num_delay_tokens,
            dtype=model.language_model.model.norm.weight.dtype,
        )
        language_token = model.tokenizer.language_token_id(language)
        self.token_queue = deque(
            [model.tokenizer.bos_token_id, language_token]
            + [model.tokenizer.streaming_pad_token_id] * (self.prefill_tokens - 2)
        )
        self._pcm = np.zeros(self.left_pad_tokens * self.samples_per_token, dtype=np.float32)
        self._pcm_start = 0
        self._clock_start = 0
        self._clock_end = self.prefill_tokens * self.samples_per_token
        self.generated_token_ids: list[int] = []
        self.semantic_vad_probabilities: list[np.ndarray] = []
        self.finalized = False

    def feed(self, pcm: mx.array | np.ndarray) -> tuple[int, ...]:
        if self.finalized:
            raise RuntimeError("cannot feed a finalized Audio8 stream")
        wave = np.asarray(pcm, dtype=np.float32)
        if wave.ndim != 1:
            raise ValueError(f"Expected mono PCM [samples], got {wave.shape}")
        if wave.shape[0]:
            self._pcm = np.concatenate([self._pcm, wave])
        return self._process_ready()

    def finalize(self) -> tuple[int, ...]:
        if self.finalized:
            return ()
        self.finalized = True
        self._pcm = np.concatenate(
            [self._pcm, np.zeros(self.right_pad_tokens * self.samples_per_token, dtype=np.float32)]
        )
        return self._process_ready()

    def result(self) -> Audio8Result:
        generated = tuple(self.generated_token_ids)
        return Audio8Result(
            text=self.model.tokenizer.decode_visible(generated),
            tokens=tuple(self.model.tokenizer.visible_token_ids(generated)),
            language=self.language,
            generated_token_ids=generated,
            semantic_vad_probabilities=tuple(self.semantic_vad_probabilities),
        )

    def _process_ready(self) -> tuple[int, ...]:
        emitted = []
        while (window := self._peek_window()) is not None:
            token_id = self._decode_window(window)
            self._commit_window(window.clock_end)
            emitted.extend(self.model.tokenizer.visible_token_ids([token_id]))
        return tuple(emitted)

    def _peek_window(self) -> _AudioWindow | None:
        frame_start = max(self._clock_start - self.look_back_samples, 0)
        frame_end = self._clock_end + self.look_ahead_samples
        available = self._pcm_start + self._pcm.shape[0]
        if frame_end > available:
            return None
        token_count = (self._clock_end - self._clock_start) // self.samples_per_token
        return _AudioWindow(
            audio=self._pcm[
                frame_start - self._pcm_start : frame_end - self._pcm_start
            ],
            token_count=token_count,
            clock_end=self._clock_end,
        )

    def _commit_window(self, clock_end: int) -> None:
        self._clock_start = clock_end
        self._clock_end = clock_end + self.samples_per_token
        keep_from = max(self._clock_start - self.look_back_samples, 0)
        trim = keep_from - self._pcm_start
        if trim > 0:
            self._pcm = self._pcm[trim:]
            self._pcm_start = keep_from

    def _decode_window(self, window: _AudioWindow) -> int:
        expected_frames = window.token_count * self.frame_len
        if (
            not self.rolling_context_tokens
            and self.audio_cache.position_offset + expected_frames
            > self.model.config.audio_config.max_position_embeddings
        ):
            raise RuntimeError(
                "Audio8 native audio context is exhausted; enable rolling_context_tokens"
            )
        features = self.model.feature_extractor(window.audio, center=True)
        conv_embeds = self.model.audio_tower.embedder(features)
        if conv_embeds.shape[1] < expected_frames:
            raise RuntimeError("Audio8 frontend produced too few audio frames")
        conv_embeds = conv_embeds[:, -expected_frames:, :]
        audio_output = self.model.audio_tower(
            inputs_embeds=conv_embeds, cache=self.audio_cache
        )
        audio_embeds = self.model.project_audio(
            audio_output.last_hidden_state, frame_len=self.frame_len
        )
        prompt = [self.token_queue.popleft() for _ in range(window.token_count)]
        input_ids = mx.array([prompt], dtype=mx.int32)
        inputs_embeds = self.model.build_text_inputs_embeds(
            input_ids=input_ids, audio_embeds=audio_embeds
        )
        output = self.model.language_model(
            inputs_embeds=inputs_embeds,
            t_cond=self.t_cond,
            cache=self.text_cache,
            logits_to_keep=1,
        )
        self.text_cache = output.cache
        logits = output.logits[0, -1].astype(mx.float32)
        logits[self.model.tokenizer.eos_token_id] = float("-inf")
        token_id = int(mx.argmax(logits).item())
        self.generated_token_ids.append(token_id)
        self.token_queue.append(token_id)
        if self.model.semantic_vad_heads:
            probabilities = self.model.semantic_vad(output.last_hidden_state[:, -1:, :])
            # Copy: a view would pin its MLX buffer for the life of the session.
            self.semantic_vad_probabilities.append(np.array(probabilities[0, :, 0, :]))
        self._trim_rolling_context()
        cache_arrays = []
        for layer in self.audio_cache.layers:
            cache_arrays.extend((layer.keys, layer.values, layer.positions))
        for layer in self.text_cache:
            cache_arrays.extend((layer.keys, layer.values))
        mx.eval(*cache_arrays)
        return token_id

    def _trim_rolling_context(self) -> None:
        if not self.rolling_context_tokens or self.text_cache is None:
            return
        if text_cache_length(self.text_cache) > self.rolling_context_tokens:
            trim_text_cache(
                self.text_cache,
                stable_prefix=self.stable_prefix_tokens,
                max_tokens=self.rolling_trim_target,
                inv_freq=self.model.language_model.model.rotary_emb._inv_freq(),
            )
        if self.audio_cache.position_offset >= self.model.config.audio_config.max_position_embeddings:
            rebase_audio_cache(
                self.audio_cache,
                target_position=self.model.config.audio_config.sliding_window,
                inv_freq=self.model.audio_tower.rotary_emb.inv_freq,
            )
