from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import FishAudioDecoderConfig, FishTextConfig


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    return mx.repeat(x, repeats, axis=1)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class FishRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float):
        super().__init__()
        self.dim = dim
        self.base = float(base)

    def __call__(self, seq_len: int, *, offset: int = 0) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.dim, 2, dtype=mx.float32)
                / mx.array(self.dim, dtype=mx.float32)
            )
        )
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def apply(
        self,
        query: mx.array,
        key: mx.array,
        *,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        cos, sin = self(seq_len=int(query.shape[2]), offset=offset)
        cos = cos[None, None, :, :].astype(query.dtype)
        sin = sin[None, None, :, :].astype(query.dtype)
        return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (
            _rotate_half(key) * sin
        )


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class FishSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        rope_base: float,
        attention_qkv_bias: bool,
        attention_o_bias: bool,
        attention_qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        if n_head % n_local_heads != 0:
            raise ValueError(
                "n_head must be divisible by n_local_heads: "
                f"{n_head} vs {n_local_heads}"
            )

        self.dim = dim
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.kv_repeat = n_head // n_local_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_size = n_head * head_dim
        self.kv_size = n_local_heads * head_dim
        self.wqkv = nn.Linear(
            dim,
            self.q_size + (2 * self.kv_size),
            bias=attention_qkv_bias,
        )
        self.wo = nn.Linear(n_head * head_dim, dim, bias=attention_o_bias)
        if attention_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=norm_eps)
        self.rotary_emb = FishRotaryEmbedding(head_dim, base=rope_base)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = mx.split(
            qkv,
            [self.q_size, self.q_size + self.kv_size],
            axis=-1,
        )
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb.apply(q, k)
        k = _repeat_kv(k, self.kv_repeat)
        v = _repeat_kv(v, self.kv_repeat)

        mask = None
        if seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                seq_len, dtype=x.dtype
            )
            mask = mask[None, None, :, :]

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.n_head * self.head_dim
        )
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        rope_base: float,
        norm_eps: float,
        attention_qkv_bias: bool,
        attention_o_bias: bool,
        attention_qk_norm: bool,
    ):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attention = FishSelfAttention(
            dim=dim,
            n_head=n_head,
            n_local_heads=n_local_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            attention_qkv_bias=attention_qkv_bias,
            attention_o_bias=attention_o_bias,
            attention_qk_norm=attention_qk_norm,
            norm_eps=norm_eps,
        )
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim, intermediate_size)

    @classmethod
    def from_text_config(cls, config: FishTextConfig) -> "TransformerBlock":
        return cls(
            dim=config.dim,
            intermediate_size=config.intermediate_size,
            n_head=config.n_head,
            n_local_heads=config.n_local_heads,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            norm_eps=config.norm_eps,
            attention_qkv_bias=config.attention_qkv_bias,
            attention_o_bias=config.attention_o_bias,
            attention_qk_norm=config.attention_qk_norm,
        )

    @classmethod
    def from_audio_config(cls, config: FishAudioDecoderConfig) -> "TransformerBlock":
        return cls(
            dim=config.dim,
            intermediate_size=config.intermediate_size,
            n_head=config.n_head,
            n_local_heads=config.n_local_heads,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
            norm_eps=config.norm_eps,
            attention_qkv_bias=config.attention_qkv_bias,
            attention_o_bias=config.attention_o_bias,
            attention_qk_norm=config.attention_qk_norm,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))
