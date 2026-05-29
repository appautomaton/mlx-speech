"""Granite Speech Conformer CTC encoder."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import GraniteSpeechEncoderConfig


class BatchNorm1d(nn.Module):
    """Inference BatchNorm1d for channel-last sequence tensors."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(self.running_var + self.eps) * self.weight + self.bias


class ConformerFeedForward(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim * config.feedforward_mult,
        )
        self.down_proj = nn.Linear(
            config.hidden_dim * config.feedforward_mult,
            config.hidden_dim,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_norm(x)
        x = nn.silu(self.up_proj(x))
        return self.down_proj(x)


class ConformerAttention(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = config.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = self.pre_norm(x)
        batch, seq_len, _ = x.shape

        num_blocks = math.ceil(seq_len / self.context_size)
        remainder = seq_len % self.context_size
        if remainder > 0:
            pad_len = self.context_size - remainder
            x = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])

        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(batch, num_blocks, self.context_size, self.num_heads, -1)
        k = k.reshape(batch, num_blocks, self.context_size, self.num_heads, -1)
        v = v.reshape(batch, num_blocks, self.context_size, self.num_heads, -1)

        q = q.transpose(0, 1, 3, 2, 4)
        k = k.transpose(0, 1, 3, 2, 4)
        v = v.transpose(0, 1, 3, 2, 4)

        rel_pos_emb = self.rel_pos_emb(attention_dists)
        pos_attn = (
            mx.sum(
                q[:, :, :, :, None, :] * rel_pos_emb[None, None, None, :, :, :],
                axis=-1,
            )
            * self.scale
        )

        if remainder > 0:
            context = self.context_size
            row_valid = mx.arange(context)[:, None] < remainder
            col_valid = mx.arange(context)[None, :] < remainder
            mask = ~(row_valid & col_valid)
            mask_value = mx.array(mx.finfo(pos_attn.dtype).min)
            pos_attn_last = mx.where(
                mask[None, None, None],
                mask_value,
                pos_attn[:, -1:, :, :, :],
            )
            pos_attn = mx.concatenate([pos_attn[:, :-1, :, :, :], pos_attn_last], axis=1)

        attn_weights = (q @ k.transpose(0, 1, 2, 4, 3)) * self.scale + pos_attn
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(q.dtype)

        out = attn_weights @ v
        out = out.transpose(0, 1, 3, 2, 4)
        out = out.reshape(batch, -1, self.num_heads * self.dim_head)
        out = out[:, :seq_len, :]
        return self.to_out(out)


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.padding[0], self.padding[1]), (0, 0)])
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.depth_conv = DepthWiseConv1d(inner_dim, config.conv_kernel_size)
        self.batch_norm = BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.up_conv(x)
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)
        x = self.depth_conv(x)
        x = nn.silu(self.batch_norm(x))
        return self.down_conv(x)


class ConformerBlock(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config)
        self.attn = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, attention_dists) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        return self.post_norm(x)


class GraniteSpeechEncoder(nn.Module):
    """Conformer CTC encoder used before the QFormer projector."""

    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = [ConformerBlock(config) for _ in range(config.num_layers)]
        self.out = nn.Linear(config.hidden_dim, config.output_dim)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim)
        self.num_layers = config.num_layers

        seq = mx.arange(config.context_size)
        relpos_dist = seq[:, None] - seq[None, :]
        self._attention_dists = (
            mx.clip(relpos_dist, -config.context_size, config.context_size)
            + config.max_pos_emb
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(f"Expected encoder input [B, T, C], got shape {x.shape}")
        x = self.input_linear(x)
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, attention_dists=self._attention_dists)
            if idx == self.num_layers // 2:
                x_mid = self.out(x)
                x = x + self.out_mid(mx.softmax(x_mid.astype(mx.float32), axis=-1).astype(x.dtype))
        return x
