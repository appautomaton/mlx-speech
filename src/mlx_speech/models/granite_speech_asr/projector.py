"""Granite Speech QFormer projector."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import GraniteSpeechConfig, GraniteSpeechProjectorConfig


class QFormerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_hidden_size: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kv_dim = kv_hidden_size or hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(kv_dim, hidden_size)
        self.value = nn.Linear(kv_dim, hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        q = self.query(hidden_states)
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.key(kv_input)
        v = self.value(kv_input)

        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = (q * (self.head_dim**-0.5)) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return out


class QFormerSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class QFormerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_hidden_size: int | None = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = QFormerMultiHeadAttention(
            hidden_size,
            num_heads,
            kv_hidden_size=kv_hidden_size,
        )
        self.output = QFormerSelfOutput(hidden_size, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        attn_out = self.attention(hidden_states, encoder_hidden_states)
        return self.output(attn_out, hidden_states)


class QFormerIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.dense(x))


class QFormerOutput(nn.Module):
    def __init__(self, intermediate_size: int, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class QFormerLayer(nn.Module):
    def __init__(self, config: GraniteSpeechProjectorConfig):
        super().__init__()
        self.attention = QFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            eps=config.layer_norm_eps,
        )
        self.crossattention = QFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            kv_hidden_size=config.encoder_hidden_size,
            eps=config.layer_norm_eps,
        )
        self.intermediate_query = QFormerIntermediate(
            config.hidden_size,
            config.intermediate_size,
        )
        self.output_query = QFormerOutput(
            config.intermediate_size,
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        hidden_states = self.attention(hidden_states)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states)
        intermediate = self.intermediate_query(hidden_states)
        return self.output_query(intermediate, hidden_states)


class QFormerEncoder(nn.Module):
    def __init__(self, config: GraniteSpeechProjectorConfig):
        super().__init__()
        self.layer = [QFormerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class QFormerModel(nn.Module):
    def __init__(self, config: GraniteSpeechProjectorConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = QFormerEncoder(config)

    def __call__(self, query_embeds: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        hidden_states = self.layernorm(query_embeds)
        return self.encoder(hidden_states, encoder_hidden_states)


class GraniteSpeechProjector(nn.Module):
    """Project encoder frames into Granite LM hidden states."""

    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.hidden_size = config.projector.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate
        self.query = mx.zeros((1, self.num_queries, config.projector.hidden_size))
        self.qformer = QFormerModel(config.projector)
        self.linear = nn.Linear(config.projector.hidden_size, config.text.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, length, dim = hidden_states.shape
        nblocks = math.ceil(length / self.window_size)
        pad = nblocks * self.window_size - length
        if pad > 0:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, pad), (0, 0)])

        hidden_states = hidden_states.reshape(batch * nblocks, self.window_size, dim)
        query = mx.broadcast_to(
            self.query,
            (batch * nblocks, self.num_queries, self.hidden_size),
        )
        query_output = self.qformer(query, hidden_states)
        return self.linear(query_output.reshape(batch, nblocks * self.num_queries, -1))
