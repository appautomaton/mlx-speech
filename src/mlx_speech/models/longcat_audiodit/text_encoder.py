"""MLX-native UMT5 encoder used by LongCat AudioDiT."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import LongCatTextEncoderConfig


@dataclass(frozen=True)
class LongCatEncoderOutput:
    last_hidden_state: mx.array
    hidden_states: tuple[mx.array, ...]


class UMT5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class UMT5DenseGatedGeluDense(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.wo(nn.gelu(self.wi_0(hidden_states)) * self.wi_1(hidden_states))


class UMT5Attention(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.key_value_proj_dim = config.d_kv
        self.inner_dim = config.num_heads * config.d_kv
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets,
            config.num_heads,
        )

    def _relative_position_bucket(self, relative_position: mx.array) -> mx.array:
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance
        ret = (relative_position > 0).astype(mx.int32) * (num_buckets // 2)
        relative_position = mx.abs(relative_position)
        num_buckets //= 2
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position = relative_position.astype(mx.float32)
        scale = mx.log(relative_position / max_exact + 1e-6) / mx.log(
            max_distance / max_exact
        )
        large_val = max_exact + (scale * (num_buckets - max_exact)).astype(mx.int32)
        large_val = mx.minimum(
            large_val, mx.full(large_val.shape, num_buckets - 1, dtype=mx.int32)
        )
        return ret + mx.where(is_small, relative_position.astype(mx.int32), large_val)

    def _compute_bias(self, query_length: int, key_length: int) -> mx.array:
        context_position = mx.arange(query_length, dtype=mx.int32)[:, None]
        memory_position = mx.arange(key_length, dtype=mx.int32)[None, :]
        relative_position = memory_position - context_position
        bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(bucket)
        return mx.transpose(values, (2, 0, 1))[None, :, :, :]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        batch, seq_len, _ = hidden_states.shape
        head_dim = self.key_value_proj_dim

        query_states = self.q(hidden_states).reshape(
            batch, seq_len, self.num_heads, head_dim
        )
        key_states = self.k(hidden_states).reshape(
            batch, seq_len, self.num_heads, head_dim
        )
        value_states = self.v(hidden_states).reshape(
            batch, seq_len, self.num_heads, head_dim
        )

        query_states = mx.transpose(query_states, (0, 2, 1, 3))
        key_states = mx.transpose(key_states, (0, 2, 1, 3))
        value_states = mx.transpose(value_states, (0, 2, 1, 3))

        scores = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2)))
        scores = scores + self._compute_bias(seq_len, seq_len).astype(scores.dtype)

        mask = attention_mask[:, None, None, :].astype(mx.bool_)
        scores = mx.where(mask, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype))
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attn_output = mx.matmul(probs, value_states)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3)).reshape(
            batch, seq_len, self.inner_dim
        )
        return self.o(attn_output)


class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.SelfAttention = UMT5Attention(config)
        self.layer_norm = UMT5LayerNorm(config.d_model, config.layer_norm_epsilon)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        normed = self.layer_norm(hidden_states)
        return hidden_states + self.SelfAttention(normed, attention_mask)


class UMT5LayerFF(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.DenseReluDense = UMT5DenseGatedGeluDense(config)
        self.layer_norm = UMT5LayerNorm(config.d_model, config.layer_norm_epsilon)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        normed = self.layer_norm(hidden_states)
        return hidden_states + self.DenseReluDense(normed)


class UMT5Block(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.layer = [
            UMT5LayerSelfAttention(config),
            UMT5LayerFF(config),
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        hidden_states = self.layer[0](hidden_states, attention_mask)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states


class UMT5EncoderStack(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = [UMT5Block(config) for _ in range(config.num_layers)]
        self.final_layer_norm = UMT5LayerNorm(config.d_model, config.layer_norm_epsilon)

    def __call__(
        self,
        *,
        input_ids: mx.array,
        attention_mask: mx.array,
        output_hidden_states: bool = False,
    ) -> LongCatEncoderOutput:
        hidden_states = self.embed_tokens(input_ids)
        collected: list[mx.array] = [hidden_states] if output_hidden_states else []
        for block in self.block:
            hidden_states = block(hidden_states, attention_mask)
            if output_hidden_states:
                collected.append(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states and collected:
            collected[-1] = hidden_states
        return LongCatEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(collected),
        )


class LongCatUMT5Encoder(nn.Module):
    def __init__(self, config: LongCatTextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = UMT5EncoderStack(config)

    def __call__(
        self,
        *,
        input_ids: mx.array,
        attention_mask: mx.array,
        output_hidden_states: bool = False,
    ) -> LongCatEncoderOutput:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
