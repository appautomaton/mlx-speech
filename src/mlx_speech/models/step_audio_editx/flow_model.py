"""Non-stream CosyVoice flow model support for Step-Audio-EditX."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...checkpoints import load_torch_archive_state_dict
from .flow import (
    PreparedStepAudioNonStreamInputs,
    load_step_audio_flow_conditioner,
)
from .frontend import resolve_step_audio_cosyvoice_dir


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _leaky_relu(x: mx.array, negative_slope: float = 0.01) -> mx.array:
    return mx.maximum(x, 0) + float(negative_slope) * mx.minimum(x, 0)


def _mish(x: mx.array) -> mx.array:
    return x * mx.tanh(mx.logaddexp(x, mx.zeros_like(x)))


class StepAudioLayerNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.weight = mx.ones((dim,), dtype=mx.float32) if affine else None
        self.bias = mx.zeros((dim,), dtype=mx.float32) if affine else None

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        centered = x32 - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight
        if self.bias is not None:
            normalized = normalized + self.bias
        return normalized.astype(x.dtype)


class StepAudioConv1d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels), dtype=mx.float32)
        self.bias = mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.transpose(0, 2, 1)
        y = mx.conv1d(
            x_nlc.astype(mx.float32),
            self.weight.astype(mx.float32),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.bias is not None:
            y = y + self.bias.astype(mx.float32)
        return y.transpose(0, 2, 1).astype(x.dtype)


class StepAudioLinearNoSubsampling(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.norm = StepAudioLayerNorm(output_size, eps=1e-5, affine=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.linear(x))


class StepAudioEspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = int(d_model)
        self.xscale = math.sqrt(float(d_model))
        self.max_len = int(max_len)
        self._pe = self._build_pe(self.max_len)

    def _build_pe(self, max_len: int) -> np.ndarray:
        position = np.arange(max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(math.log(10000.0) / float(self.d_model))
        )
        pe_positive = np.zeros((max_len, self.d_model), dtype=np.float32)
        pe_negative = np.zeros((max_len, self.d_model), dtype=np.float32)
        pe_positive[:, 0::2] = np.sin(position * div_term)
        pe_positive[:, 1::2] = np.cos(position * div_term)
        pe_negative[:, 0::2] = np.sin(-1.0 * position * div_term)
        pe_negative[:, 1::2] = np.cos(-1.0 * position * div_term)
        pe_positive = np.flip(pe_positive, axis=0)[None, :, :]
        pe_negative = pe_negative[1:][None, :, :]
        pe = np.concatenate([pe_positive, pe_negative], axis=1)
        return pe

    def _ensure_size(self, size: int) -> None:
        needed = size * 2 - 1
        if int(self._pe.shape[1]) >= needed:
            return
        new_max = max(size, self.max_len * 2)
        self.max_len = int(new_max)
        self._pe = self._build_pe(self.max_len)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        self._ensure_size(int(x.shape[1]))
        x = x * self.xscale
        pos_emb = self.position_encoding(size=int(x.shape[1]))
        return x, pos_emb

    def position_encoding(self, *, size: int) -> mx.array:
        self._ensure_size(size)
        center = int(self._pe.shape[1] // 2)
        return mx.array(self._pe[:, center - size + 1 : center + size, :], dtype=mx.float32)


def _make_pad_mask(lengths: mx.array | np.ndarray, max_len: int = 0) -> mx.array:
    lengths_np = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if lengths_np.size == 0:
        raise ValueError("Pad-mask lengths must not be empty.")
    resolved_max_len = int(max_len) if int(max_len) > 0 else int(lengths_np.max())
    seq = np.arange(resolved_max_len, dtype=np.int64)[None, :]
    return mx.array(seq >= lengths_np[:, None])


class StepAudioRelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, *, key_bias: bool = True):
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError(f"Expected n_feat divisible by n_head, got {n_feat} and {n_head}.")
        self.h = int(n_head)
        self.d_k = int(n_feat // n_head)
        self.linear_q = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = mx.zeros((self.h, self.d_k), dtype=mx.float32)
        self.pos_bias_v = mx.zeros((self.h, self.d_k), dtype=mx.float32)

    def _forward_qkv(self, query: mx.array, key: mx.array, value: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch_size = int(query.shape[0])
        q = self.linear_q(query).reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = self.linear_k(key).reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        v = self.linear_v(value).reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        return q, k, v

    def _rel_shift(self, x: mx.array) -> mx.array:
        zero_pad = mx.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype)
        x_padded = mx.concatenate([zero_pad, x], axis=-1)
        x_padded = x_padded.reshape(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        shifted = x_padded[:, :, 1:].reshape(x.shape)
        return shifted[:, :, :, : x.shape[-1] // 2 + 1]

    def _masked_softmax(self, scores: mx.array, mask: mx.array) -> mx.array:
        if mask.size == 0:
            return mx.softmax(scores, axis=-1)
        if mask.ndim == 3 and mask.shape[1] == 1:
            invalid = mx.logical_not(mask)[:, :, None, :]
        elif mask.ndim == 3:
            invalid = mx.logical_not(mask)[:, None, :, :]
        else:
            raise ValueError(f"Unsupported attention mask shape: {mask.shape}.")
        neg_inf = mx.zeros_like(scores) - 1e9
        masked_scores = mx.where(invalid, neg_inf, scores)
        attn = mx.softmax(masked_scores, axis=-1)
        return mx.where(invalid, mx.zeros_like(attn), attn)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
    ) -> mx.array:
        q, k, v = self._forward_qkv(query, key, value)
        q_t = q.transpose(0, 2, 1, 3)

        p = self.linear_pos(pos_emb).reshape(pos_emb.shape[0], -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        q_with_bias_u = (q_t + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_with_bias_v = (q_t + self.pos_bias_v).transpose(0, 2, 1, 3)

        matrix_ac = mx.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))
        matrix_bd = mx.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))
        matrix_bd = self._rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(float(self.d_k))
        attn = self._masked_softmax(scores, mask)
        x = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], self.h * self.d_k)
        return self.linear_out(x)


class StepAudioPositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int):
        super().__init__()
        self.linear1 = nn.Linear(idim, hidden_units, bias=True)
        self.linear2 = nn.Linear(hidden_units, idim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(_silu(self.linear1(x)))


class StepAudioConformerEncoderLayer(nn.Module):
    def __init__(self, size: int, *, attention_heads: int, linear_units: int, key_bias: bool):
        super().__init__()
        self.self_attn = StepAudioRelPositionMultiHeadedAttention(
            attention_heads,
            size,
            key_bias=key_bias,
        )
        self.feed_forward = StepAudioPositionwiseFeedForward(size, linear_units)
        self.norm_ff = StepAudioLayerNorm(size, eps=1e-12, affine=True)
        self.norm_mha = StepAudioLayerNorm(size, eps=1e-12, affine=True)
        self.ff_scale = 1.0

    def __call__(self, x: mx.array, mask: mx.array, pos_emb: mx.array) -> mx.array:
        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, mask, pos_emb)
        x = residual + x_att

        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        return x


class StepAudioUpsample1D(nn.Module):
    def __init__(self, channels: int, out_channels: int, *, stride: int, scale_factor: float):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.scale_factor = float(scale_factor)
        self.conv = StepAudioConv1d(out_channels, channels, self.stride * 2 + 1, bias=True)

    def __call__(self, inputs: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        repeated = mx.repeat(inputs, repeats=int(self.scale_factor), axis=2)
        padded = mx.pad(repeated, ((0, 0), (0, 0), (self.stride * 2, 0)))
        outputs = self.conv(padded)
        return outputs, input_lengths * self.stride


class StepAudioPreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, *, pre_lookahead_len: int):
        super().__init__()
        self.channels = int(channels)
        self.pre_lookahead_len = int(pre_lookahead_len)
        self.conv1 = StepAudioConv1d(
            channels,
            channels,
            self.pre_lookahead_len + 1,
            bias=True,
        )
        self.conv2 = StepAudioConv1d(channels, channels, 3, bias=True)

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs = inputs.transpose(0, 2, 1)
        outputs = mx.pad(outputs, ((0, 0), (0, 0), (0, self.pre_lookahead_len)))
        outputs = _leaky_relu(self.conv1(outputs))
        outputs = mx.pad(outputs, ((0, 0), (0, 0), (2, 0)))
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(0, 2, 1)
        return outputs + inputs


class StepAudioUpsampleConformerEncoderV2(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        pre_lookahead_len: int,
        num_blocks: int,
        num_up_blocks: int,
        up_stride: int,
        up_scale_factor: float,
        attention_heads: int,
        linear_units: int,
        key_bias: bool,
    ):
        super().__init__()
        self.embed = StepAudioLinearNoSubsampling(input_size, output_size)
        self.embed_pos = StepAudioEspnetRelPositionalEncoding(output_size)
        self.normalize_before = True
        self.after_norm = StepAudioLayerNorm(output_size, eps=1e-5, affine=True)
        self.pre_lookahead_layer = StepAudioPreLookaheadLayer(
            channels=output_size,
            pre_lookahead_len=pre_lookahead_len,
        )
        self.encoders = [
            StepAudioConformerEncoderLayer(
                output_size,
                attention_heads=attention_heads,
                linear_units=linear_units,
                key_bias=key_bias,
            )
            for _ in range(num_blocks)
        ]
        self.up_layer = StepAudioUpsample1D(
            channels=output_size,
            out_channels=output_size,
            stride=up_stride,
            scale_factor=up_scale_factor,
        )
        self.up_embed = StepAudioLinearNoSubsampling(input_size, output_size)
        self.up_embed_pos = StepAudioEspnetRelPositionalEncoding(output_size)
        self.up_encoders = [
            StepAudioConformerEncoderLayer(
                output_size,
                attention_heads=attention_heads,
                linear_units=linear_units,
                key_bias=key_bias,
            )
            for _ in range(num_up_blocks)
        ]

    def __call__(self, xs: mx.array, xs_lens: mx.array) -> tuple[mx.array, mx.array]:
        time = int(xs.shape[1])
        masks = mx.logical_not(_make_pad_mask(xs_lens, time))[:, None, :]
        xs = self.embed(xs)
        xs, pos_emb = self.embed_pos(xs)
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            xs = layer(xs, masks, pos_emb)

        xs = xs.transpose(0, 2, 1)
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(0, 2, 1)

        time = int(xs.shape[1])
        masks = mx.logical_not(_make_pad_mask(xs_lens, time))[:, None, :]
        xs = self.up_embed(xs)
        xs, pos_emb = self.up_embed_pos(xs)
        for layer in self.up_encoders:
            xs = layer(xs, masks, pos_emb)
        xs = self.after_norm(xs)
        return xs, masks


class StepAudioMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        return self.fc2(x)


class StepAudioDiTAttention(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=True)
        self.q_norm = StepAudioLayerNorm(self.head_dim, eps=1e-5, affine=True)
        self.k_norm = StepAudioLayerNorm(self.head_dim, eps=1e-5, affine=True)
        self.proj = nn.Linear(self.inner_dim, dim, bias=True)

    def _to_heads(self, x: mx.array) -> mx.array:
        batch, time, channels = x.shape
        return x.reshape(batch, time, self.num_heads, channels // self.num_heads).transpose(0, 2, 1, 3)

    def _masked_softmax(self, scores: mx.array, attn_mask: mx.array | None) -> mx.array:
        if attn_mask is None:
            return mx.softmax(scores, axis=-1)
        if attn_mask.ndim == 3:
            invalid = mx.logical_not(attn_mask)[:, None, :, :]
        elif attn_mask.ndim == 4:
            invalid = mx.logical_not(attn_mask)
        else:
            raise ValueError(f"Unsupported DiT attention mask shape: {attn_mask.shape}.")
        neg_inf = mx.zeros_like(scores) - 1e9
        masked_scores = mx.where(invalid, neg_inf, scores)
        attn = mx.softmax(masked_scores, axis=-1)
        return mx.where(invalid, mx.zeros_like(attn), attn)

    def __call__(self, x: mx.array, attn_mask: mx.array | None) -> mx.array:
        batch, time, _ = x.shape
        q = self._to_heads(self.to_q(x))
        k = self._to_heads(self.to_k(x))
        v = self._to_heads(self.to_v(x))
        q = self.q_norm(q)
        k = self.k_norm(k)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self._masked_softmax(scores, attn_mask)
        x = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(batch, time, self.inner_dim)
        return self.proj(x)


def _modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


class StepAudioTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.scale = 1000.0
        self.linear1 = nn.Linear(self.frequency_embedding_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def _timestep_embedding(self, t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=t.dtype) / float(half)
        )
        args = t[:, None] * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._timestep_embedding(t * self.scale, self.frequency_embedding_size)
        return self.linear2(_silu(self.linear1(t_freq)))


class StepAudioCausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.conv = StepAudioConv1d(out_channels, in_channels, kernel_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        padded = mx.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))
        return self.conv(padded)


class StepAudioCausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3):
        super().__init__()
        self.conv1 = StepAudioCausalConv1d(in_channels, out_channels, kernel_size)
        self.norm = StepAudioLayerNorm(out_channels, eps=1e-5, affine=True)
        self.conv2 = StepAudioCausalConv1d(out_channels, out_channels, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.transpose(0, 2, 1)
        y = self.conv1(y).transpose(0, 2, 1)
        y = _mish(self.norm(y))
        y = y.transpose(0, 2, 1)
        y = self.conv2(y).transpose(0, 2, 1)
        return y


class StepAudioDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, head_dim: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = StepAudioLayerNorm(hidden_size, eps=1e-6, affine=False)
        self.attn = StepAudioDiTAttention(hidden_size, num_heads=num_heads, head_dim=head_dim)
        self.norm2 = StepAudioLayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mlp = StepAudioMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.norm3 = StepAudioLayerNorm(hidden_size, eps=1e-6, affine=False)
        self.conv = StepAudioCausalConvBlock(hidden_size, hidden_size, kernel_size=3)
        self.adaLN_linear = nn.Linear(hidden_size, hidden_size * 9, bias=True)

    def __call__(self, x: mx.array, c: mx.array, attn_mask: mx.array | None) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = mx.split(
            self.adaLN_linear(_silu(c)),
            9,
            axis=-1,
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_conv * self.conv(_modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class StepAudioFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.adaLN_linear = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.norm_final = StepAudioLayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        shift, scale = mx.split(self.adaLN_linear(_silu(c)), 2, axis=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class StepAudioDiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.t_embedder = StepAudioTimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.blocks = [
            StepAudioDiTBlock(
                hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ]
        self.final_layer = StepAudioFinalLayer(hidden_size, out_channels)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array | None,
        cond: mx.array | None,
    ) -> mx.array:
        timestep = self.t_embedder(t).reshape(t.shape[0], 1, -1)
        parts = [x, mu]
        if spks is not None:
            repeated = mx.broadcast_to(spks[:, :, None], (spks.shape[0], spks.shape[1], x.shape[-1]))
            parts.append(repeated)
        if cond is not None:
            parts.append(cond)
        merged = mx.concatenate(parts, axis=1).transpose(0, 2, 1)
        hidden = self.in_proj(merged)
        attn_mask = mask.astype(mx.bool_)
        for block in self.blocks:
            hidden = block(hidden, timestep, attn_mask)
        return self.final_layer(hidden, timestep).transpose(0, 2, 1)


class StepAudioCausalConditionalCFM(nn.Module):
    def __init__(self, estimator: StepAudioDiT, *, inference_cfg_rate: float):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.out_channels = estimator.out_channels
        rng = np.random.default_rng(seed=0)
        self._rand_noise = rng.standard_normal((1, self.out_channels, 50 * 600), dtype=np.float32)

    def _solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
    ) -> mx.array:
        t = t_span[:1]
        dt = t_span[1:2] - t
        mask_in = mx.concatenate([mask, mask], axis=0)
        mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
        spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
        cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)
        batch_size = int(x.shape[0])

        for step in range(1, int(t_span.shape[0])):
            x_in = mx.concatenate([x, x], axis=0)
            t_in = mx.concatenate([t, t], axis=0)
            dphi = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt = dphi[:batch_size]
            cfg_dphi_dt = dphi[batch_size:]
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt) - (
                self.inference_cfg_rate * cfg_dphi_dt
            )
            x = x + dt.reshape(1, 1, 1) * dphi_dt
            t = t + dt
            if step < int(t_span.shape[0]) - 1:
                dt = t_span[step + 1 : step + 2] - t
        return x

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        *,
        n_timesteps: int = 10,
        temperature: float = 1.0,
    ) -> mx.array:
        rand_noise = mx.array(self._rand_noise[:, :, : mu.shape[2]], dtype=mu.dtype)
        z = rand_noise * float(temperature)
        t_span = mx.linspace(0.0, 1.0, int(n_timesteps) + 1, dtype=mu.dtype)
        t_span = 1.0 - mx.cos(t_span * 0.5 * math.pi)
        return self._solve_euler(z, t_span, mu, mask, spks, cond)


class StepAudioCausalMaskedDiffWithXvec(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        spk_embed_dim: int,
        vocab_size: int,
        encoder: StepAudioUpsampleConformerEncoderV2,
        decoder: StepAudioCausalConditionalCFM,
        input_embedding: nn.Module,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.vocab_size = int(vocab_size)
        self.input_embedding = input_embedding
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size, bias=True)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.after_norm.weight.shape[0], output_size, bias=True)
        self.decoder = decoder

    def inference(
        self,
        prepared: PreparedStepAudioNonStreamInputs,
        *,
        n_timesteps: int = 10,
    ) -> np.ndarray:
        token = mx.array(prepared.token_dual, dtype=mx.int64)
        prompt_token = mx.array(prepared.prompt_token_dual, dtype=mx.int64)
        prompt_feat = mx.array(prepared.prompt_feat_aligned, dtype=mx.float32)
        embedding = mx.array(prepared.normalized_speaker_embedding, dtype=mx.float32)

        embedding = self.spk_embed_affine_layer(embedding)
        token_len = mx.array([int(token.shape[1])], dtype=mx.int64)
        prompt_token_len = mx.array([int(prompt_token.shape[1])], dtype=mx.int64)
        token_len = prompt_token_len + token_len
        token = mx.concatenate([prompt_token, token], axis=1)

        valid_mask = mx.logical_not(_make_pad_mask(token_len, int(token.shape[1])))[..., None]
        token_embed = self.input_embedding(mx.maximum(token, 0)) * valid_mask.astype(mx.float32)

        h, _ = self.encoder(token_embed, token_len)
        h = self.encoder_proj(h)

        mel_len1 = int(prompt_feat.shape[1])
        mel_len2 = int(h.shape[1] - prompt_feat.shape[1])
        conds = mx.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(0, 2, 1)

        total_mel_len = mx.array([mel_len1 + mel_len2], dtype=mx.int64)
        mask = mx.logical_not(_make_pad_mask(total_mel_len, mel_len1 + mel_len2))
        feat = self.decoder(
            h.transpose(0, 2, 1),
            mask[:, None, :],
            embedding,
            conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]
        if int(feat.shape[2]) != mel_len2:
            raise ValueError(
                f"Expected generated mel length {mel_len2}, got {int(feat.shape[2])}."
            )
        return np.asarray(feat, dtype=np.float32)


@dataclass(frozen=True)
class StepAudioFlowRuntimeConfig:
    input_size: int
    output_size: int
    spk_embed_dim: int
    vocab_size: int
    encoder_output_size: int
    pre_lookahead_len: int
    num_blocks: int
    num_up_blocks: int
    up_stride: int
    up_scale_factor: float
    attention_heads: int
    linear_units: int
    key_bias: bool
    estimator_in_channels: int
    estimator_out_channels: int
    estimator_hidden_size: int
    estimator_depth: int
    estimator_num_heads: int
    estimator_head_dim: int
    estimator_mlp_ratio: float
    inference_cfg_rate: float = 0.7

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, mx.array]) -> "StepAudioFlowRuntimeConfig":
        encoder_block_ids = sorted(
            {
                int(match.group(1))
                for key in state_dict
                if (match := re.match(r"encoder\.encoders\.(\d+)\.", key))
            }
        )
        up_block_ids = sorted(
            {
                int(match.group(1))
                for key in state_dict
                if (match := re.match(r"encoder\.up_encoders\.(\d+)\.", key))
            }
        )
        estimator_block_ids = sorted(
            {
                int(match.group(1))
                for key in state_dict
                if (match := re.match(r"decoder\.estimator\.blocks\.(\d+)\.", key))
            }
        )
        input_embedding = state_dict["input_embedding.embedding.weight"]
        spk_proj = state_dict["spk_embed_affine_layer.weight"]
        encoder_proj = state_dict["encoder_proj.weight"]
        pos_bias = state_dict["encoder.encoders.0.self_attn.pos_bias_u"]
        feed_forward = state_dict["encoder.encoders.0.feed_forward.w_1.weight"]
        up_conv = state_dict["encoder.up_layer.conv.weight"]
        estimator_in_proj = state_dict["decoder.estimator.in_proj.weight"]
        final_linear = state_dict["decoder.estimator.final_layer.linear.weight"]
        estimator_q = state_dict["decoder.estimator.blocks.0.attn.to_q.weight"]
        estimator_mlp = state_dict["decoder.estimator.blocks.0.mlp.fc1.weight"]

        return cls(
            input_size=int(input_embedding.shape[1] * 2),
            output_size=int(encoder_proj.shape[0]),
            spk_embed_dim=int(spk_proj.shape[1]),
            vocab_size=int(input_embedding.shape[0]),
            encoder_output_size=int(encoder_proj.shape[1]),
            pre_lookahead_len=int(state_dict["encoder.pre_lookahead_layer.conv1.weight"].shape[2] - 1),
            num_blocks=(encoder_block_ids[-1] + 1) if encoder_block_ids else 0,
            num_up_blocks=(up_block_ids[-1] + 1) if up_block_ids else 0,
            up_stride=int((up_conv.shape[2] - 1) // 2),
            up_scale_factor=float((up_conv.shape[2] - 1) // 2),
            attention_heads=int(pos_bias.shape[0]),
            linear_units=int(feed_forward.shape[0]),
            key_bias="encoder.encoders.0.self_attn.linear_k.bias" in state_dict,
            estimator_in_channels=int(estimator_in_proj.shape[1]),
            estimator_out_channels=int(final_linear.shape[0]),
            estimator_hidden_size=int(estimator_in_proj.shape[0]),
            estimator_depth=(estimator_block_ids[-1] + 1) if estimator_block_ids else 0,
            estimator_num_heads=int(estimator_q.shape[0] // state_dict["decoder.estimator.blocks.0.attn.q_norm.weight"].shape[0]),
            estimator_head_dim=int(state_dict["decoder.estimator.blocks.0.attn.q_norm.weight"].shape[0]),
            estimator_mlp_ratio=float(estimator_mlp.shape[0] / estimator_in_proj.shape[0]),
        )


@dataclass(frozen=True)
class StepAudioFlowCheckpoint:
    model_dir: Path
    config: StepAudioFlowRuntimeConfig
    state_dict: dict[str, mx.array]


@dataclass(frozen=True)
class StepAudioFlowAlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return (
            not self.missing_in_model
            and not self.missing_in_checkpoint
            and not self.shape_mismatches
        )


@dataclass(frozen=True)
class LoadedStepAudioFlowModel:
    model_dir: Path
    config: StepAudioFlowRuntimeConfig
    checkpoint: StepAudioFlowCheckpoint
    model: StepAudioCausalMaskedDiffWithXvec
    alignment_report: StepAudioFlowAlignmentReport


def _map_checkpoint_key(key: str) -> str:
    mapped = key
    mapped = mapped.replace("encoder.embed.out.0.", "encoder.embed.linear.")
    mapped = mapped.replace("encoder.embed.out.1.", "encoder.embed.norm.")
    mapped = mapped.replace("encoder.up_embed.out.0.", "encoder.up_embed.linear.")
    mapped = mapped.replace("encoder.up_embed.out.1.", "encoder.up_embed.norm.")
    mapped = mapped.replace(".feed_forward.w_1.", ".feed_forward.linear1.")
    mapped = mapped.replace(".feed_forward.w_2.", ".feed_forward.linear2.")
    mapped = mapped.replace("decoder.estimator.t_embedder.mlp.0.", "decoder.estimator.t_embedder.linear1.")
    mapped = mapped.replace("decoder.estimator.t_embedder.mlp.2.", "decoder.estimator.t_embedder.linear2.")
    mapped = mapped.replace(".conv.block.1.", ".conv.conv1.conv.")
    mapped = mapped.replace(".conv.block.3.", ".conv.norm.")
    mapped = mapped.replace(".conv.block.6.", ".conv.conv2.conv.")
    mapped = mapped.replace(".adaLN_modulation.1.", ".adaLN_linear.")
    mapped = mapped.replace("decoder.estimator.final_layer.adaLN_modulation.1.", "decoder.estimator.final_layer.adaLN_linear.")
    return mapped


def sanitize_step_audio_flow_state_dict(
    state_dict: dict[str, mx.array],
) -> tuple[StepAudioFlowRuntimeConfig, dict[str, mx.array]]:
    config = StepAudioFlowRuntimeConfig.from_state_dict(state_dict)
    sanitized: dict[str, mx.array] = {}
    for key, value in state_dict.items():
        mapped_key = _map_checkpoint_key(key)
        mapped_value = value
        if key in {
            "encoder.pre_lookahead_layer.conv1.weight",
            "encoder.pre_lookahead_layer.conv2.weight",
            "encoder.up_layer.conv.weight",
        } or re.match(r"decoder\.estimator\.blocks\.\d+\.conv\.block\.[16]\.weight$", key):
            mapped_value = value.transpose(0, 2, 1)
        sanitized[mapped_key] = mapped_value
    return config, sanitized


def load_step_audio_flow_checkpoint(model_dir: str | Path) -> StepAudioFlowCheckpoint:
    resolved_model_dir = resolve_step_audio_cosyvoice_dir(model_dir)
    archive = load_torch_archive_state_dict(resolved_model_dir / "flow.pt")
    config, state_dict = sanitize_step_audio_flow_state_dict(archive.weights)
    return StepAudioFlowCheckpoint(
        model_dir=resolved_model_dir,
        config=config,
        state_dict=state_dict,
    )


def validate_step_audio_flow_checkpoint_against_model(
    model: StepAudioCausalMaskedDiffWithXvec,
    checkpoint: StepAudioFlowCheckpoint,
) -> StepAudioFlowAlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioFlowAlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def _load_flow_model_from_safetensors(
    model_dir: Path,
) -> LoadedStepAudioFlowModel:
    import json

    config_path = model_dir / "flow-model-config.json"
    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    quantization = payload.pop("quantization", None)
    config = StepAudioFlowRuntimeConfig(**payload)

    estimator = StepAudioDiT(
        in_channels=config.estimator_in_channels,
        out_channels=config.estimator_out_channels,
        mlp_ratio=config.estimator_mlp_ratio,
        depth=config.estimator_depth,
        num_heads=config.estimator_num_heads,
        head_dim=config.estimator_head_dim,
        hidden_size=config.estimator_hidden_size,
    )
    decoder = StepAudioCausalConditionalCFM(
        estimator, inference_cfg_rate=config.inference_cfg_rate,
    )
    encoder = StepAudioUpsampleConformerEncoderV2(
        input_size=config.input_size,
        output_size=config.encoder_output_size,
        pre_lookahead_len=config.pre_lookahead_len,
        num_blocks=config.num_blocks,
        num_up_blocks=config.num_up_blocks,
        up_stride=config.up_stride,
        up_scale_factor=config.up_scale_factor,
        attention_heads=config.attention_heads,
        linear_units=config.linear_units,
        key_bias=config.key_bias,
    )
    conditioner = load_step_audio_flow_conditioner(model_dir)
    model = StepAudioCausalMaskedDiffWithXvec(
        input_size=config.input_size,
        output_size=config.output_size,
        spk_embed_dim=config.spk_embed_dim,
        vocab_size=config.vocab_size,
        encoder=encoder,
        decoder=decoder,
        input_embedding=conditioner.model.input_embedding,
    )
    if quantization is not None:
        import mlx.nn as nn

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda p, m: (
                hasattr(m, "weight") and hasattr(m, "to_quantized")
                and m.weight.shape[-1] % quantization["group_size"] == 0
            ),
        )
    weights = mx.load(str(model_dir / "flow-model.safetensors"))
    model.load_weights(list(weights.items()))
    return LoadedStepAudioFlowModel(
        model_dir=model_dir,
        config=config,
        checkpoint=StepAudioFlowCheckpoint(
            model_dir=model_dir, config=config, state_dict=weights,
        ),
        model=model,
        alignment_report=StepAudioFlowAlignmentReport(
            missing_in_model=(), missing_in_checkpoint=(), shape_mismatches=(),
        ),
    )


def load_step_audio_flow_model(
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> LoadedStepAudioFlowModel:
    resolved = Path(model_dir)
    if (resolved / "flow-model.safetensors").exists():
        return _load_flow_model_from_safetensors(resolved)
    checkpoint = load_step_audio_flow_checkpoint(model_dir)
    config = checkpoint.config
    estimator = StepAudioDiT(
        in_channels=config.estimator_in_channels,
        out_channels=config.estimator_out_channels,
        mlp_ratio=config.estimator_mlp_ratio,
        depth=config.estimator_depth,
        num_heads=config.estimator_num_heads,
        head_dim=config.estimator_head_dim,
        hidden_size=config.estimator_hidden_size,
    )
    decoder = StepAudioCausalConditionalCFM(
        estimator,
        inference_cfg_rate=config.inference_cfg_rate,
    )
    encoder = StepAudioUpsampleConformerEncoderV2(
        input_size=config.input_size,
        output_size=config.encoder_output_size,
        pre_lookahead_len=config.pre_lookahead_len,
        num_blocks=config.num_blocks,
        num_up_blocks=config.num_up_blocks,
        up_stride=config.up_stride,
        up_scale_factor=config.up_scale_factor,
        attention_heads=config.attention_heads,
        linear_units=config.linear_units,
        key_bias=config.key_bias,
    )
    conditioner = load_step_audio_flow_conditioner(model_dir)
    model = StepAudioCausalMaskedDiffWithXvec(
        input_size=config.input_size,
        output_size=config.output_size,
        spk_embed_dim=config.spk_embed_dim,
        vocab_size=config.vocab_size,
        encoder=encoder,
        decoder=decoder,
        input_embedding=conditioner.model.input_embedding,
    )
    report = validate_step_audio_flow_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio flow checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return LoadedStepAudioFlowModel(
        model_dir=checkpoint.model_dir,
        config=config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
    )


__all__ = [
    "LoadedStepAudioFlowModel",
    "StepAudioCausalConditionalCFM",
    "StepAudioCausalMaskedDiffWithXvec",
    "StepAudioFlowAlignmentReport",
    "StepAudioFlowCheckpoint",
    "StepAudioFlowRuntimeConfig",
    "StepAudioUpsampleConformerEncoderV2",
    "load_step_audio_flow_checkpoint",
    "load_step_audio_flow_model",
    "sanitize_step_audio_flow_state_dict",
    "validate_step_audio_flow_checkpoint_against_model",
]
