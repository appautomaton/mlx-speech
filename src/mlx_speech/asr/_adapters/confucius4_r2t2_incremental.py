"""Incremental R2T2 streaming step.

Each window the full-refeed step recomputes mel, the audio tower, and the
decoder prompt over all audio seen so far. This step keeps per-session state
and recomputes only what can have changed:

- log-mel: raw frames whose STFT window lies inside the audio are final and
  kept; only right-edge frames are recomputed. The clip-global floor
  (``max - 8``) is recomputed over all frames every window, as offline.
- audio tower: attention runs in independent blocks of 8 conv chunks (800
  frames). A closed block whose frames are final and whose clipping is
  unaffected by the current floor is reused; the first dirty block and
  everything after it are recomputed.
- decoder: the KV cache keeps every position before the first changed
  embedding (first differing token id, or the first recomputed audio row);
  only the suffix is prefilled, at the correct RoPE offset.

R2T2 semantics (prompt, schedule, greedy decode, budget) are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ...generation.qwen3_asr import validate_context_window
from ...models.qwen3_asr import _get_feat_extract_output_lengths
from ...models.qwen3_asr.audio_encoder import _activation, _attention_block_lengths
from ...models.qwen3_asr.feature_extraction import _get_mel_filters, _hann_window
from ...models.qwen3_asr.text_decoder import Qwen3ASRTextKVCache
from ...models.confucius4_r2t2.metrics import (
    CACHE_ALLOCATIONS,
    ENCODER_BLOCKS,
    ENCODER_TOKENS,
    MEL_FRAMES,
    PREFILL_POSITIONS,
    VOCAB_POSITIONS,
    R2T2StepMetrics,
    stage,
)

_INITIAL_KV = 1024


@dataclass
class _Block:
    features: mx.array  # encoder output rows for one closed block
    floor: float  # mel floor used when it was encoded
    raw_min: float  # smallest raw log-mel value in the block


class R2T2IncrementalStep:
    """One session's reusable mel, audio-block, and KV state."""

    def __init__(self, runtime, *, context: str, language: str | None, compute) -> None:
        self._runtime = runtime
        self._context = context
        self._language = language
        self._compute = compute
        extractor = runtime.processor.feature_extractor
        if float(extractor.dither) != 0.0:
            raise ValueError("Incremental R2T2 streaming requires dither == 0.")
        self._hop = int(extractor.hop_length)
        self._n_fft = int(extractor.n_fft)
        self._window = _hann_window(self._n_fft).astype(np.float64)
        self._filters = _get_mel_filters(
            sample_rate=int(extractor.sample_rate),
            n_fft=self._n_fft,
            n_mels=int(extractor.n_mels),
        )
        # raw log-mel buffer: [:, :_stable] final, the rest scratch
        self._raw = np.zeros((int(extractor.n_mels), 4096), dtype=np.float32)
        self._stable = 0  # raw frames [0, _stable) are final
        self._stable_max = -np.inf

        tower = runtime.model.audio_tower
        self._chunk = int(tower.n_window) * 2
        multiplier = max(1, int(tower.n_window_infer) // self._chunk)
        self._block_frames = self._chunk * multiplier
        self._block_tokens = int(_get_feat_extract_output_lengths(self._chunk)) * multiplier
        self._blocks: list[_Block] = []

        self._kv: Qwen3ASRTextKVCache | None = None
        self._kv_ids: list[int] = []

    # ------------------------------------------------------------------ mel
    def _update_raw(self, audio: np.ndarray, frames: int) -> np.ndarray:
        """Raw log-mel for frames [0, frames); keeps the final ones."""

        samples = int(audio.shape[0])
        half = self._n_fft // 2
        first = self._stable
        lo = first * self._hop - half
        if lo < 0:
            padded = np.pad(audio.astype(np.float64), [(half, half)], mode="reflect")
            offset = first * self._hop
        else:
            padded = np.pad(audio[lo:].astype(np.float64), [(0, half)], mode="reflect")
            offset = 0
        count = frames - first
        windows = np.lib.stride_tricks.sliding_window_view(padded, self._n_fft)
        windows = windows[offset : offset + count * self._hop : self._hop][:count]
        spectrum = (np.abs(np.fft.rfft(windows * self._window, axis=1)) ** 2).astype(np.float32)
        mel = np.maximum(1e-10, self._filters.T @ spectrum.T)
        fresh = np.log10(mel).astype(np.float32)
        if frames > self._raw.shape[1]:
            grown = np.zeros((self._raw.shape[0], max(frames, 2 * self._raw.shape[1])), dtype=np.float32)
            grown[:, :first] = self._raw[:, :first]
            self._raw = grown
        self._raw[:, first:frames] = fresh
        raw = self._raw[:, :frames]

        stable = min(frames, (samples - half) // self._hop + 1) if samples >= half else 0
        if stable > first:
            self._stable_max = max(self._stable_max, float(raw[:, first:stable].max()))
            self._stable = stable
        return raw

    # -------------------------------------------------------------- encoder
    def _encode(self, normalized: np.ndarray, *, pad_width: int, max_after: int) -> mx.array:
        model = self._runtime.model
        tower = model.audio_tower
        dtype = mx.bfloat16 if self._compute.audio_bf16 else mx.float32
        features = mx.array(normalized, dtype=dtype)
        frames = int(normalized.shape[1])
        chunked = tower._chunk_features(features, frames)
        padded = chunked.padded_features
        if int(padded.shape[-1]) < pad_width:
            padded = mx.pad(padded, [(0, 0), (0, 0), (0, pad_width - int(padded.shape[-1]))])
        hidden = tower._conv_forward(padded, chunked.aftercnn_lengths)
        block_lengths = _attention_block_lengths(
            total_length=int(sum(chunked.aftercnn_lengths)),
            max_chunk_aftercnn=max_after,
            n_window=tower.n_window,
            n_window_infer=tower.n_window_infer,
        )
        for layer in tower.layers:
            hidden = layer(hidden, block_lengths=block_lengths)
        hidden = tower.ln_post(hidden)
        hidden = tower.proj1(hidden)
        hidden = _activation(hidden, tower.activation_function)
        return tower.proj2(hidden)

    def _audio_features(
        self, audio: np.ndarray, metrics: R2T2StepMetrics | None
    ) -> tuple[mx.array, int, int]:
        """Encoder rows for all audio, and the first recomputed row."""

        frames = int(audio.shape[0]) // self._hop
        if frames <= 0:
            raise ValueError("Confucius4-R2T2 streaming needs at least one mel frame.")
        with stage(metrics, "mel"):
            first_new = self._stable
            raw = self._update_raw(audio, frames)
            tail_max = float(raw[:, self._stable :].max()) if frames > self._stable else -np.inf
            # float32 arithmetic, exactly as the offline extractor
            floor = np.float32(max(self._stable_max, tail_max)) - np.float32(8.0)

        with stage(metrics, "audio_tower"):
            dirty = 0
            if frames >= self._chunk:
                for index, block in enumerate(self._blocks):
                    end = (index + 1) * self._block_frames
                    if end > self._stable:
                        break
                    if block.floor != float(floor) and block.raw_min < max(float(floor), block.floor):
                        break
                    dirty = index + 1
            del self._blocks[dirty:]
            start = dirty * self._block_frames
            normalized = (np.maximum(raw[:, start:frames], floor) + 4.0) / 4.0
            pad_width = min(self._chunk, frames)
            max_after = int(_get_feat_extract_output_lengths(pad_width))
            fresh = self._encode(normalized, pad_width=pad_width, max_after=max_after)
            mx.eval(fresh)
            if frames >= self._chunk:
                closed = min(frames, self._stable) // self._block_frames
                for index in range(dirty, closed):
                    row = (index - dirty) * self._block_tokens
                    lo = index * self._block_frames
                    self._blocks.append(
                        _Block(
                            features=fresh[row : row + self._block_tokens],
                            floor=float(floor),
                            raw_min=float(raw[:, lo : lo + self._block_frames].min()),
                        )
                    )
            reused = [block.features for block in self._blocks[:dirty]]
            features = mx.concatenate([*reused, fresh], axis=0) if reused else fresh

        if metrics is not None:
            metrics.mel_frames = frames
            metrics.add_counter(MEL_FRAMES, frames - first_new)
            metrics.add_counter(ENCODER_TOKENS, int(fresh.shape[0]))
            metrics.add_counter(
                ENCODER_BLOCKS,
                max(1, -(-int(fresh.shape[0]) // self._block_tokens)),
            )
        return features, dirty * self._block_tokens, frames

    # -------------------------------------------------------------- decoder
    def prefill(
        self,
        *,
        prompt_ids,
        audio: np.ndarray,
        max_new_tokens: int,
        metrics: R2T2StepMetrics | None,
    ):
        """Update caches and prefill the changed suffix. Returns the output
        whose ``logits`` are the last-position logits and the prompt ids."""

        runtime = self._runtime
        model = runtime.model
        compute = self._compute
        features, dirty_row, _ = self._audio_features(audio, metrics)
        audio_tokens = int(features.shape[0])

        with stage(metrics, "prompt_build"):
            ids = prompt_ids(audio_tokens)
        validate_context_window(
            prompt_tokens=len(ids),
            max_new_tokens=max_new_tokens,
            max_position_embeddings=runtime.config.text_config.max_position_embeddings,
        )
        audio_id = int(runtime.config.audio_token_id)
        a0 = ids.index(audio_id)
        if ids[a0 : a0 + audio_tokens] != [audio_id] * audio_tokens:
            raise ValueError("Unexpected R2T2 prompt layout: audio span is not contiguous.")

        common = 0
        for left, right in zip(ids, self._kv_ids):
            if left != right:
                break
            common += 1
        keep = min(common, a0 + dirty_row, len(ids) - 1)

        embed_dtype = mx.bfloat16 if compute.residual_bf16 else model.text_decoder.model.embed_tokens.weight.dtype
        if self._kv is None or self._kv.layers[0].dtype != embed_dtype:
            limit = int(runtime.config.text_config.max_position_embeddings)
            self._kv = Qwen3ASRTextKVCache.allocate(
                runtime.config.text_config,
                batch_size=1,
                max_length=min(limit, max(len(ids) + max_new_tokens, _INITIAL_KV)),
                dtype=embed_dtype,
            )
            self._kv_ids = []
            keep = 0
            if metrics is not None:
                metrics.add_counter(CACHE_ALLOCATIONS)
        keep = min(keep, self._kv.current_length)
        self._kv.truncate(keep)

        with stage(metrics, "embed"):
            tokens = model.embed_input_ids(mx.array([ids[keep:]], dtype=mx.int32))
            s0 = max(a0, keep) - keep
            s1 = a0 + audio_tokens - keep
            if s1 > s0:
                rows = features[max(a0, keep) - a0 :][None].astype(tokens.dtype)
                embeds = mx.concatenate([tokens[:, :s0], rows, tokens[:, s1:]], axis=1)
            else:
                embeds = tokens
            embeds = embeds.astype(embed_dtype)
        limit = int(runtime.config.text_config.max_position_embeddings)
        if metrics is not None:
            metrics.audio_tokens = audio_tokens
        return self._run(ids, embeds, keep, len(ids) + max_new_tokens, limit, metrics)

    def _run(self, ids, embeds, keep, needed, limit, metrics):
        kv = self._kv
        if needed > kv.max_length:
            kv.reserve(min(limit, max(needed, 2 * kv.max_length)))
        with stage(metrics, "prefill_body"):
            output = self._runtime.model.text_decoder.decode_step(
                inputs_embeds=embeds,
                kv_cache=kv,
                compute=self._compute.decoder,
                last_logits_only=True,
            )
            mx.eval(output.logits)
        kv.prompt_length = len(ids)
        self._kv_ids = list(ids)
        if metrics is not None:
            metrics.prompt_tokens = len(ids)
            metrics.add_counter(PREFILL_POSITIONS, len(ids) - keep)
            metrics.add_counter(VOCAB_POSITIONS, 1)
        return output
