"""MLX audio encoder structure for Qwen3-ASR."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Qwen3ASRAudioConfig
from .feature_extraction import _get_feat_extract_output_lengths


@dataclass(frozen=True)
class Qwen3ASRAudioEncoderOutput:
    """Audio tower output consumed by multimodal fusion."""

    last_hidden_state: mx.array


class Qwen3ASRSinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("Qwen3-ASR sinusoidal embedding requires an even channel count.")
        log_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = np.exp(-log_increment * np.arange(channels // 2, dtype=np.float32))
        scaled_time = np.arange(length, dtype=np.float32)[:, None] * inv_timescales[None, :]
        self.positional_embedding = mx.array(
            np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1),
            dtype=mx.float32,
        )

    def __call__(self, seq_len: int) -> mx.array:
        if seq_len > self.positional_embedding.shape[0]:
            raise ValueError(
                f"Qwen3-ASR audio position length {seq_len} exceeds "
                f"max_source_positions {self.positional_embedding.shape[0]}."
            )
        return self.positional_embedding[:seq_len, :]


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config: Qwen3ASRAudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Qwen3-ASR audio d_model {self.embed_dim} must be divisible by "
                f"encoder_attention_heads {self.num_heads}."
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        block_lengths: Sequence[int] | None = None,
    ) -> mx.array:
        if hidden_states.ndim != 2:
            raise ValueError(f"Expected audio attention input [T, C], got {hidden_states.shape}.")
        if hidden_states.shape[0] == 0:
            return hidden_states

        block_lengths = _normalize_block_lengths(block_lengths, total_length=hidden_states.shape[0])
        outputs = [
            self._attend_block(hidden_states[start:end])
            for start, end in _iter_blocks(block_lengths)
            if end > start
        ]
        return mx.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    def _attend_block(self, x: mx.array) -> mx.array:
        seq_len = x.shape[0]
        q = self.q_proj(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = self.k_proj(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = self.v_proj(x).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        scores = (q @ k.transpose(0, 2, 1)) * self.scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = weights @ v
        out = out.transpose(1, 0, 2).reshape(seq_len, self.embed_dim)
        return self.out_proj(out)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASRAudioConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.activation_function = config.activation_function

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        block_lengths: Sequence[int] | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, block_lengths=block_lengths)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = _activation(hidden_states, self.activation_function)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    """Qwen3-ASR audio tower with Conv2D downsampling and transformer layers."""

    def __init__(self, config: Qwen3ASRAudioConfig):
        super().__init__()
        self.config = config
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.positional_embedding = Qwen3ASRSinusoidalPositionEmbedding(
            config.max_source_positions,
            config.d_model,
        )
        self.layers = [Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            stride=2,
            padding=1,
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * _conv2d_subsampled_length(config.num_mel_bins),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.activation_function = config.activation_function

    def __call__(
        self,
        input_features: mx.array,
        *,
        feature_lens: int | Sequence[int] | None = None,
    ) -> Qwen3ASRAudioEncoderOutput:
        features = mx.array(input_features)
        if features.ndim == 3:
            if features.shape[0] != 1:
                raise ValueError("Qwen3-ASR audio encoder v0 supports one audio input at a time.")
            features = features[0]
        if features.ndim != 2:
            raise ValueError(f"Expected Qwen3-ASR audio features [mel, frames], got {features.shape}.")
        if features.shape[0] != self.num_mel_bins:
            raise ValueError(
                f"Expected {self.num_mel_bins} mel bins, got {features.shape[0]}."
            )

        feature_len = _coerce_single_length(
            feature_lens,
            default=features.shape[1],
            max_value=features.shape[1],
        )
        chunked = self._chunk_features(features[:, :feature_len], feature_len)
        hidden_states = self._conv_forward(
            chunked.padded_features,
            chunked.aftercnn_lengths,
        )
        block_lengths = _attention_block_lengths(
            total_length=int(sum(chunked.aftercnn_lengths)),
            max_chunk_aftercnn=max(chunked.aftercnn_lengths),
            n_window=self.n_window,
            n_window_infer=self.n_window_infer,
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, block_lengths=block_lengths)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = _activation(hidden_states, self.activation_function)
        hidden_states = self.proj2(hidden_states)
        return Qwen3ASRAudioEncoderOutput(last_hidden_state=hidden_states)

    def output_lengths(self, feature_lens: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        return _get_feat_extract_output_lengths(feature_lens)

    def _chunk_features(self, features: mx.array, feature_len: int) -> "_ChunkedFeatures":
        chunk_size = self.n_window * 2
        chunk_lengths = [min(chunk_size, feature_len - start) for start in range(0, feature_len, chunk_size)]
        if not chunk_lengths:
            raise ValueError("Qwen3-ASR audio encoder requires non-empty feature frames.")

        max_chunk_len = max(chunk_lengths)
        chunks = []
        for start, length in zip(range(0, feature_len, chunk_size), chunk_lengths, strict=True):
            chunk = features[:, start : start + length]
            if length < max_chunk_len:
                chunk = mx.pad(chunk, [(0, 0), (0, max_chunk_len - length)])
            chunks.append(chunk)

        aftercnn_lengths = [
            int(_get_feat_extract_output_lengths(length))
            for length in chunk_lengths
        ]
        return _ChunkedFeatures(
            padded_features=mx.stack(chunks, axis=0),
            chunk_lengths=chunk_lengths,
            aftercnn_lengths=aftercnn_lengths,
        )

    def _conv_forward(
        self,
        padded_features: mx.array,
        aftercnn_lengths: Sequence[int],
    ) -> mx.array:
        x = padded_features[:, :, :, None]
        conv_outputs = []
        for chunk in _split_first_axis(x, self.conv_chunksize):
            chunk = nn.gelu(self.conv2d1(chunk))
            chunk = nn.gelu(self.conv2d2(chunk))
            chunk = nn.gelu(self.conv2d3(chunk))
            conv_outputs.append(chunk)
        x = mx.concatenate(conv_outputs, axis=0) if len(conv_outputs) > 1 else conv_outputs[0]

        batch, freq_out, time_out, channels = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(batch, time_out, channels * freq_out)
        x = self.conv_out(x)
        x = x + self.positional_embedding(time_out)[None, :, :].astype(x.dtype)

        valid_chunks = [
            x[index, : int(length), :]
            for index, length in enumerate(aftercnn_lengths)
            if int(length) > 0
        ]
        if not valid_chunks:
            raise ValueError("Qwen3-ASR audio encoder produced zero audio tokens.")
        return mx.concatenate(valid_chunks, axis=0) if len(valid_chunks) > 1 else valid_chunks[0]


@dataclass(frozen=True)
class _ChunkedFeatures:
    padded_features: mx.array
    chunk_lengths: list[int]
    aftercnn_lengths: list[int]


def _activation(x: mx.array, name: str) -> mx.array:
    if name == "gelu":
        return nn.gelu(x)
    if name == "silu":
        return nn.silu(x)
    raise ValueError(f"Unsupported Qwen3-ASR audio activation: {name}")


def _conv2d_subsampled_length(length: int) -> int:
    out = int(length)
    for _ in range(3):
        out = (out - 1) // 2 + 1
    return out


def _coerce_single_length(
    value: int | Sequence[int] | None,
    *,
    default: int,
    max_value: int,
) -> int:
    if value is None:
        out = int(default)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError("Qwen3-ASR audio encoder v0 accepts one feature length.")
        out = int(value[0])
    else:
        out = int(value)
    if out <= 0:
        raise ValueError("Qwen3-ASR audio encoder requires a positive feature length.")
    if out > max_value:
        raise ValueError(f"feature_lens {out} exceeds input feature frames {max_value}.")
    return out


def _attention_block_lengths(
    *,
    total_length: int,
    max_chunk_aftercnn: int,
    n_window: int,
    n_window_infer: int,
) -> list[int]:
    window_multiplier = max(1, n_window_infer // (n_window * 2))
    window_aftercnn = max(1, max_chunk_aftercnn * window_multiplier)
    lengths: list[int] = []
    remaining = int(total_length)
    while remaining > 0:
        block = min(window_aftercnn, remaining)
        lengths.append(block)
        remaining -= block
    return lengths


def _normalize_block_lengths(
    block_lengths: Sequence[int] | None,
    *,
    total_length: int,
) -> list[int]:
    if block_lengths is None:
        return [int(total_length)]
    lengths = [int(length) for length in block_lengths if int(length) > 0]
    if sum(lengths) != total_length:
        raise ValueError(
            f"Qwen3-ASR audio attention block lengths sum to {sum(lengths)}, "
            f"expected {total_length}."
        )
    return lengths


def _iter_blocks(lengths: Sequence[int]):
    start = 0
    for length in lengths:
        end = start + int(length)
        yield start, end
        start = end


def _split_first_axis(x: mx.array, chunk_size: int) -> list[mx.array]:
    chunk_size = max(1, int(chunk_size))
    return [x[start : start + chunk_size] for start in range(0, x.shape[0], chunk_size)]
