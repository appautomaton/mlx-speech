"""True cache-aware waveform streaming for Nemotron 3.5 ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import PreprocessArgs
from .feature_extraction import _MEL_COMPUTE_BLOCK
from .prompt import apply_language_prompt
from .transducer import LSTMState

if TYPE_CHECKING:
    from .encoder import FastConformerEncoder
    from .model import NemotronASRModel, NemotronASRResult

_SUBSAMPLING_HISTORY = 16


class FixedFrameCache:
    """Fixed-capacity mirrored ring with a contiguous logical cache view."""

    def __init__(
        self,
        capacity: int,
        width: int,
        *,
        dtype: mx.Dtype,
        initially_full: bool = False,
    ) -> None:
        if capacity < 0:
            raise ValueError("cache capacity must be non-negative")
        self.capacity = capacity
        self.buffer = mx.zeros((1, 2, capacity, width), dtype=dtype)
        self.length = capacity if initially_full else 0
        self.write_offset = 0

    def _flat(self) -> mx.array:
        return self.buffer.reshape(1, self.capacity * 2, self.buffer.shape[-1])

    def values(self) -> mx.array:
        flat = self._flat()
        if self.length < self.capacity:
            return flat[:, : self.length]
        return flat[:, self.write_offset : self.write_offset + self.capacity]

    def joined(self, *tails: mx.array) -> mx.array:
        """Return ordered cached frames plus tails without an intermediate copy."""
        pieces = []
        if self.length:
            pieces.append(self.values())
        pieces.extend(tail for tail in tails if tail.shape[1])
        if not pieces:
            return self.buffer[:, :0]
        if len(pieces) == 1:
            return pieces[0]
        return mx.concatenate(pieces, axis=1)

    def append(self, frames: mx.array) -> None:
        if frames.ndim != 3 or frames.shape[0] != 1:
            raise ValueError(f"expected cache frames [1, T, D], got {frames.shape}")
        count = frames.shape[1]
        if count == 0 or self.capacity == 0:
            return
        if count >= self.capacity:
            tail = frames[:, -self.capacity :]
            self.buffer[:, :] = tail[:, None]
            self.length = self.capacity
            self.write_offset = 0
            return
        first = min(count, self.capacity - self.write_offset)
        self.buffer[
            :,
            :,
            self.write_offset : self.write_offset + first,
        ] = frames[:, None, :first]
        remaining = count - first
        if remaining:
            self.buffer[:, :, :remaining] = frames[:, None, first:]
        self.write_offset = (self.write_offset + count) % self.capacity
        self.length = min(self.capacity, self.length + count)


class StreamingMelFrontend:
    """Incremental centered-STFT frontend with bounded PCM history."""

    def __init__(self, preprocessor: nn.Module, args: PreprocessArgs) -> None:
        self.preprocessor = preprocessor
        self.args = args
        self.total_samples = 0
        self.next_frame = 0
        self.buffer_start = 0
        self._samples = mx.zeros((0,), dtype=mx.float32)
        self._previous_raw: mx.array | None = None
        self.finalized = False

    @property
    def residual_sample_count(self) -> int:
        return self.total_samples % self.args.hop_length

    @property
    def buffered_sample_count(self) -> int:
        return self._samples.shape[0]

    def feed(self, pcm: mx.array | np.ndarray) -> mx.array:
        if self.finalized:
            raise RuntimeError("cannot feed a finalized mel stream")
        raw = mx.array(pcm, dtype=mx.float32)
        if raw.ndim != 1:
            raise ValueError(f"expected mono PCM [samples], got {raw.shape}")
        if raw.shape[0] == 0:
            return mx.zeros((1, 0, self.args.features), dtype=mx.float32)

        if self.args.preemph > 0.0:
            if self._previous_raw is None:
                first = raw[:1]
            else:
                first = raw[:1] - self.args.preemph * self._previous_raw
            emphasized = mx.concatenate(
                [first, raw[1:] - self.args.preemph * raw[:-1]], axis=0
            )
        else:
            emphasized = raw
        self._previous_raw = raw[-1:]
        self._samples = mx.concatenate([self._samples, emphasized], axis=0)
        self.total_samples += raw.shape[0]
        return self._emit_ready(final=False)

    def finalize(self) -> mx.array:
        if self.finalized:
            return mx.zeros((1, 0, self.args.features), dtype=mx.float32)
        self.finalized = True
        return self._emit_ready(final=True)

    def _target_frame_count(self, *, final: bool) -> int:
        if final:
            return self.total_samples // self.args.hop_length
        center = self.args.n_fft // 2
        if self.total_samples < center:
            return 0
        ready = (self.total_samples - center) // self.args.hop_length + 1
        ready = min(ready, self.total_samples // self.args.hop_length)
        return ready // _MEL_COMPUTE_BLOCK * _MEL_COMPUTE_BLOCK

    def _emit_ready(self, *, final: bool) -> mx.array:
        output_end = self._target_frame_count(final=final)
        # Offline centered STFT always computes one additional frame and masks
        # it. Include that frame in the final fixed FFT block, then discard it,
        # so the valid tail has identical kernel geometry.
        compute_end = output_end + int(final)
        if compute_end <= self.next_frame:
            return mx.zeros((1, 0, self.args.features), dtype=mx.float32)

        frame_start = self.next_frame
        count = compute_end - frame_start
        hop = self.args.hop_length
        n_fft = self.args.n_fft
        center = n_fft // 2
        sample_start = frame_start * hop - center
        sample_end = (compute_end - 1) * hop - center + n_fft
        raw_start = max(sample_start, 0)
        raw_end = min(sample_end, self.total_samples)
        local_start = raw_start - self.buffer_start
        local_end = raw_end - self.buffer_start
        segment = self._samples[local_start:local_end]
        left_pad = max(-sample_start, 0)
        right_pad = max(sample_end - self.total_samples, 0)
        if left_pad or right_pad:
            segment = mx.pad(segment, ((left_pad, right_pad),))

        expected = (count - 1) * hop + n_fft
        if segment.shape[0] != expected:
            raise RuntimeError(
                f"streaming STFT segment has {segment.shape[0]} samples, "
                f"expected {expected}"
            )
        frames = mx.as_strided(
            segment,
            shape=(count, n_fft),
            strides=(hop, 1),
        )
        featurizer = self.preprocessor.featurizer
        left = (n_fft - self.args.win_length) // 2
        right = n_fft - self.args.win_length - left
        window = mx.pad(featurizer.window, ((left, right),))
        output = featurizer.log_mel_frames(frames, window=window)
        valid_count = output_end - frame_start
        output = output[:valid_count][None]

        self.next_frame = output_end
        keep_from = max(self.next_frame * hop - center, 0)
        trim = keep_from - self.buffer_start
        if trim > 0:
            self._samples = self._samples[trim:]
            self.buffer_start = keep_from
        mx.eval(output, self._samples, self._previous_raw)
        return output.astype(mx.float32)


@dataclass
class LayerStreamingState:
    attention: FixedFrameCache
    convolution: FixedFrameCache


class StreamingEncoder:
    """Incremental FastConformer with fixed per-layer attention/conv caches."""

    def __init__(
        self,
        encoder: FastConformerEncoder,
        *,
        att_context_size: tuple[int, int],
    ) -> None:
        self.encoder = encoder
        self.left_context, self.right_context = (
            int(att_context_size[0]),
            int(att_context_size[1]),
        )
        self.chunk_frames = self.right_context + 1
        self.chunk_mel_frames = self.chunk_frames * encoder.args.subsampling_factor
        # The reference runtime keeps frontend and cached encoder activations in
        # fp32 even when checkpoint parameters are bf16.
        self.dtype = mx.float32
        self.layers = [
            LayerStreamingState(
                attention=FixedFrameCache(
                    self.left_context,
                    encoder.args.d_model,
                    dtype=self.dtype,
                ),
                convolution=FixedFrameCache(
                    encoder.args.conv_kernel_size - 1,
                    encoder.args.d_model,
                    dtype=self.dtype,
                    initially_full=True,
                ),
            )
            for _ in encoder.layers
        ]
        self.mel_history = FixedFrameCache(
            _SUBSAMPLING_HISTORY,
            encoder.args.feat_in,
            dtype=mx.float32,
        )
        self.pending_mel = mx.zeros(
            (1, 0, encoder.args.feat_in), dtype=mx.float32
        )
        self.consumed_mel_frames = 0
        self.emitted_encoder_frames = 0
        self.block_frame_evaluations = 0
        self.finalized = False

    @property
    def cache_buffers(self) -> tuple[mx.array, ...]:
        buffers = []
        for layer in self.layers:
            buffers.extend((layer.attention.buffer, layer.convolution.buffer))
        buffers.append(self.mel_history.buffer)
        return tuple(buffers)

    def feed(self, mel: mx.array, *, final: bool = False) -> list[mx.array]:
        if self.finalized:
            raise RuntimeError("cannot feed a finalized encoder stream")
        if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[2] != self.encoder.args.feat_in:
            raise ValueError(f"expected mel [1, T, {self.encoder.args.feat_in}]")
        if mel.shape[1]:
            self.pending_mel = mx.concatenate([self.pending_mel, mel], axis=1)

        output = []
        while self.pending_mel.shape[1] >= self.chunk_mel_frames:
            current = self.pending_mel[:, : self.chunk_mel_frames]
            self.pending_mel = self.pending_mel[:, self.chunk_mel_frames :]
            output.extend(self._subsample_and_encode(current, final=False))

        if final:
            tail = self.pending_mel
            self.pending_mel = self.pending_mel[:, :0]
            output.extend(self._subsample_and_encode(tail, final=True))
            self.finalized = True
        if output:
            mx.eval(*output)
        return output

    def _subsample_and_encode(
        self, mel: mx.array, *, final: bool
    ) -> list[mx.array]:
        history = self.mel_history.values()
        cache_length = history.shape[1]
        window = mel if cache_length == 0 else mx.concatenate([history, mel], axis=1)
        if window.shape[1] == 0:
            return []
        subsampled, _ = self.encoder.pre_encode(
            window,
            mx.array([window.shape[1]], dtype=mx.int32),
        )

        end = self.consumed_mel_frames + mel.shape[1]
        base = (self.consumed_mel_frames - cache_length) // self.encoder.args.subsampling_factor
        start_index = self.emitted_encoder_frames - base
        if final:
            end_index = subsampled.shape[1]
        else:
            end_index = end // self.encoder.args.subsampling_factor - base
        self.consumed_mel_frames = end
        self.mel_history.append(mel.astype(mx.float32))

        if end_index <= start_index:
            return []
        self.emitted_encoder_frames = base + end_index
        ready = subsampled[:, start_index:end_index]
        chunks = []
        for start in range(0, ready.shape[1], self.chunk_frames):
            chunk = ready[:, start : start + self.chunk_frames]
            chunks.append(self._encode_chunk(chunk))
        return chunks

    def _encode_chunk(self, hidden: mx.array) -> mx.array:
        real_frames = hidden.shape[1]
        is_native = real_frames == self.chunk_frames
        self.block_frame_evaluations += real_frames * len(self.encoder.layers)

        def native_tail(function, values):  # type: ignore[no-untyped-def]
            if real_frames == self.chunk_frames:
                return function(values)
            prefix = mx.zeros(
                (
                    values.shape[0],
                    self.chunk_frames - real_frames,
                    values.shape[2],
                ),
                dtype=values.dtype,
            )
            padded = mx.concatenate([prefix, values], axis=1)
            return function(padded)[:, -real_frames:]

        for index, block in enumerate(self.encoder.layers):
            state = self.layers[index]
            if is_native:
                residual = hidden + 0.5 * block.feed_forward1(
                    block.norm_feed_forward1(hidden)
                )
            else:
                residual = hidden + 0.5 * native_tail(
                    lambda values: block.feed_forward1(
                        block.norm_feed_forward1(values)
                    ),
                    hidden,
                )

            normalized = (
                block.norm_self_att(residual)
                if is_native
                else native_tail(block.norm_self_att, residual)
            )
            key_value = state.attention.joined(normalized)
            positions = self.encoder.pos_enc.for_length(
                key_value.shape[1], hidden.dtype
            )
            attention_query = normalized
            if not is_native:
                # Keep the fused attention query geometry identical to a native
                # chunk. Real final-tail queries occupy the trailing positions;
                # prefix outputs are discarded and never enter either cache.
                padding = mx.zeros(
                    (
                        normalized.shape[0],
                        self.chunk_frames - normalized.shape[1],
                        normalized.shape[2],
                    ),
                    dtype=normalized.dtype,
                )
                attention_query = mx.concatenate([padding, normalized], axis=1)
            attention_output = block.self_attn.stream(
                attention_query, key_value, positions
            )[:, -normalized.shape[1] :]
            residual = residual + attention_output
            state.attention.append(normalized)

            if is_native:
                convolution_input = block.norm_conv(residual)
                pointwise = block.conv.pointwise_conv1(convolution_input)
            else:
                convolution_input = native_tail(block.norm_conv, residual)
                pointwise = native_tail(
                    block.conv.pointwise_conv1, convolution_input
                )
            gated = nn.glu(pointwise, axis=-1)
            convolution_history = state.convolution.joined(gated)
            if is_native:
                convolution_window = convolution_history
            else:
                convolution_suffix = mx.zeros(
                    (
                        gated.shape[0],
                        self.chunk_frames - real_frames,
                        gated.shape[2],
                    ),
                    dtype=gated.dtype,
                )
                convolution_window = mx.concatenate(
                    [convolution_history, convolution_suffix], axis=1
                )
            convolved = block.conv.depthwise_conv(convolution_window)
            state.convolution.append(gated)
            convolved = block.conv.batch_norm(convolved)
            convolved = block.conv.activation(convolved)
            convolved = block.conv.pointwise_conv2(convolved)[:, :real_frames]
            residual = residual + convolved

            if is_native:
                residual = residual + 0.5 * block.feed_forward2(
                    block.norm_feed_forward2(residual)
                )
                hidden = block.norm_out(residual)
            else:
                residual = residual + 0.5 * native_tail(
                    lambda values: block.feed_forward2(
                        block.norm_feed_forward2(values)
                    ),
                    residual,
                )
                hidden = native_tail(block.norm_out, residual)

        return hidden


class NemotronStreamSession:
    """Persistent live waveform-to-token session."""

    def __init__(
        self,
        model: NemotronASRModel,
        *,
        language: str,
        att_context_size: tuple[int, int],
    ) -> None:
        self.model = model
        self.language = language
        self.att_context_size = att_context_size
        self.mel = StreamingMelFrontend(model.preprocessor, model.config.preprocessor)
        self.encoder = StreamingEncoder(
            model.encoder, att_context_size=att_context_size
        )
        self.tokens: list[int] = []
        self.frame_indices: list[int] = []
        self.global_frame = 0
        self.last_token = model.blank_id
        self.predictor_output, self.predictor_state = model.decoder(None, None)
        self.predictor_output = self.predictor_output[:, -1:, :]
        self.finalized = False

    def feed(self, pcm: mx.array | np.ndarray) -> tuple[int, ...]:
        if self.finalized:
            raise RuntimeError("cannot feed a finalized Nemotron stream")
        mel = self.mel.feed(pcm)
        return self._consume_encoder_chunks(self.encoder.feed(mel))

    def finalize(self) -> tuple[int, ...]:
        if self.finalized:
            return ()
        mel = self.mel.finalize()
        emitted = self._consume_encoder_chunks(self.encoder.feed(mel, final=True))
        self.finalized = True
        return emitted

    def result(self, *, strip_language_tags: bool = True) -> NemotronASRResult:
        from .model import NemotronASRResult

        token_tuple = tuple(self.tokens)
        return NemotronASRResult(
            text=self.model.tokenizer.decode(
                token_tuple, strip_language_tags=strip_language_tags
            ),
            tokens=token_tuple,
            language=self.language,
            detected_language=self.model.tokenizer.detected_language(token_tuple),
            frame_indices=tuple(self.frame_indices),
        )

    def _consume_encoder_chunks(self, chunks: list[mx.array]) -> tuple[int, ...]:
        emitted = []
        for chunk in chunks:
            prompted = apply_language_prompt(
                chunk,
                self.language,
                self.model.config.prompt,
                self.model.prompt_kernel,
            )
            emitted.extend(self._decode_chunk(prompted))
        mx.eval(self.predictor_output, self.predictor_state)
        return tuple(emitted)

    def _decode_chunk(self, encoded: mx.array) -> list[int]:
        emitted = []
        prediction = self.predictor_output.astype(encoded.dtype)
        state: LSTMState = self.predictor_state
        for local_frame in range(encoded.shape[1]):
            symbols = 0
            while True:
                logits = self.model.joint(
                    encoded[:, local_frame : local_frame + 1], prediction
                )
                token = int(mx.argmax(logits[0, 0]).item())
                if token == self.model.blank_id:
                    break

                self.tokens.append(token)
                emitted.append(token)
                self.frame_indices.append(self.global_frame + local_frame)
                self.last_token = token
                prediction, state = self.model.decoder(
                    mx.array([[token]], dtype=mx.int32), state
                )
                prediction = prediction[:, -1:, :].astype(encoded.dtype)
                symbols += 1
                if symbols >= self.model.config.max_symbols:
                    break
        self.global_frame += encoded.shape[1]
        self.predictor_output = prediction
        self.predictor_state = state
        return emitted


__all__ = [
    "FixedFrameCache",
    "LayerStreamingState",
    "NemotronStreamSession",
    "StreamingEncoder",
    "StreamingMelFrontend",
]
