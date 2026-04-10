from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .codec import validate_codec_assets
from .codec_config import FishCodecConfig


def _normalize_weight(x: mx.array, except_dim: int = 0) -> mx.array:
    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(mx.power(x, 2), axis=axes, keepdims=True))


def _conv1d_torch_to_mlx(weight: mx.array) -> mx.array:
    return weight.transpose(0, 2, 1)


def _conv_transpose1d_torch_to_mlx(weight: mx.array) -> mx.array:
    return weight.transpose(1, 2, 0)


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


def unpad1d(x: mx.array, paddings: tuple[int, int]) -> mx.array:
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = int(x.shape[-1])
    n_frames = (length - kernel_size + padding_total) / stride + 1.0
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return max(0, int(ideal_length - length))


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Conv1dTorch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        in_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_per_group * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_per_group, kernel_size),
        )
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.swapaxes(1, 2)
        y = mx.conv1d(
            x_nlc,
            _conv1d_torch_to_mlx(self.weight),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class ConvTranspose1dTorch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        out_per_group = out_channels // groups
        scale = math.sqrt(1.0 / (out_per_group * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(in_channels, out_per_group, kernel_size),
        )
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.swapaxes(1, 2)
        y = mx.conv_transpose1d(
            x_nlc,
            _conv_transpose1d_torch_to_mlx(self.weight),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class WNConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        in_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_per_group * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_per_group, kernel_size),
        )
        self.weight_g = _normalize_weight(weight_init, except_dim=0)
        self.weight_v = weight_init / (self.weight_g + 1e-12)
        self.bias = mx.zeros((out_channels,)) if bias else None

    def _weight(self) -> mx.array:
        return (
            self.weight_g
            * self.weight_v
            / _normalize_weight(self.weight_v, except_dim=0)
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.swapaxes(1, 2)
        y = mx.conv1d(
            x_nlc,
            _conv1d_torch_to_mlx(self._weight()),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class WNConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        out_per_group = out_channels // groups
        scale = math.sqrt(1.0 / (out_per_group * kernel_size))
        weight_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(in_channels, out_per_group, kernel_size),
        )
        self.weight_g = _normalize_weight(weight_init, except_dim=0)
        self.weight_v = weight_init / (self.weight_g + 1e-12)
        self.bias = mx.zeros((out_channels,)) if bias else None

    def _weight(self) -> mx.array:
        return (
            self.weight_g
            * self.weight_v
            / _normalize_weight(self.weight_v, except_dim=0)
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.swapaxes(1, 2)
        y = mx.conv_transpose1d(
            x_nlc,
            _conv_transpose1d_torch_to_mlx(self._weight()),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


def snake(x: mx.array, alpha: mx.array) -> mx.array:
    return x + mx.reciprocal(alpha + 1e-9) * mx.power(mx.sin(alpha * x), 2)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        return snake(x, self.alpha)


class CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Optional[int] = None,
    ):
        super().__init__()
        self.conv = Conv1dTorch(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def __call__(self, x: mx.array) -> mx.array:
        pad = self.padding
        extra = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, pad)
        x = mx.pad(x, [(0, 0), (0, 0), (pad, extra)])
        return self.conv(x)


class CausalTransConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Optional[int] = None,
    ):
        super().__init__()
        self.conv = ConvTranspose1dTorch(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        return unpad1d(x, (padding_left, padding_right))


class CausalWNConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = CausalConvNet(*args, **kwargs)
        w = self.conv.conv.weight
        self.weight_g = _normalize_weight(w, except_dim=0)
        self.weight_v = w / (self.weight_g + 1e-12)
        self.bias = self.conv.conv.bias
        self.conv.conv.weight = self.weight_v
        self.conv.conv.bias = self.bias

    def __call__(self, x: mx.array) -> mx.array:
        self.conv.conv.weight = (
            self.weight_g
            * self.weight_v
            / _normalize_weight(self.weight_v, except_dim=0)
        )
        self.conv.conv.bias = self.bias
        return self.conv(x)


class CausalWNConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = CausalTransConvNet(*args, **kwargs)
        w = self.conv.conv.weight
        self.weight_g = _normalize_weight(w, except_dim=0)
        self.weight_v = w / (self.weight_g + 1e-12)
        self.bias = self.conv.conv.bias
        self.conv.conv.weight = self.weight_v
        self.conv.conv.bias = self.bias

    def __call__(self, x: mx.array) -> mx.array:
        self.conv.conv.weight = (
            self.weight_g
            * self.weight_v
            / _normalize_weight(self.weight_v, except_dim=0)
        )
        self.conv.conv.bias = self.bias
        return self.conv(x)


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def __call__(self, z: mx.array):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = mx.mean(mx.power(z_e - z_q, 2), axis=[1, 2])
        codebook_loss = mx.mean(mx.power(z_q - z_e, 2), axis=[1, 2])
        z_q = z_e + (z_q - z_e)
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: mx.array) -> mx.array:
        return self.codebook.weight[embed_id]

    def decode_code(self, embed_id: mx.array) -> mx.array:
        return self.embed_code(embed_id).transpose(0, 2, 1)

    def decode_latents(self, latents: mx.array) -> tuple[mx.array, mx.array]:
        b, d, t = latents.shape
        encodings = latents.transpose(0, 2, 1).reshape(b * t, d)
        codebook = self.codebook.weight

        enc_norm = encodings / mx.maximum(
            mx.sqrt(mx.sum(mx.power(encodings, 2), axis=1, keepdims=True)), 1e-12
        )
        code_norm = codebook / mx.maximum(
            mx.sqrt(mx.sum(mx.power(codebook, 2), axis=1, keepdims=True)), 1e-12
        )
        dist = (
            mx.sum(mx.power(enc_norm, 2), axis=1, keepdims=True)
            - 2 * (enc_norm @ code_norm.T)
            + mx.sum(mx.power(code_norm, 2), axis=1, keepdims=True).T
        )
        indices = (-dist).argmax(axis=1).reshape(b, t)
        return self.decode_code(indices), indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list[int]] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout
        self.quantizers = [
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ]

    def __call__(self, z: mx.array, n_quantizers: Optional[int] = None):
        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        z_q = 0.0
        residual = z
        commitment_loss = 0.0
        codebook_loss = 0.0
        codebook_indices = []
        latents = []

        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            z_q_i, commit_i, codebk_i, indices_i, z_e_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            commitment_loss = commitment_loss + commit_i.mean()
            codebook_loss = codebook_loss + codebk_i.mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = mx.stack(codebook_indices, axis=1)
        latents = mx.concatenate(latents, axis=1)
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z_q = 0.0
        z_p = []
        n_codebooks = int(codes.shape[1])
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q = z_q + self.quantizers[i].out_proj(z_p_i)
        return z_q, mx.concatenate(z_p, axis=1), codes


@dataclass
class VQResult:
    z: mx.array
    codes: mx.array
    latents: mx.array
    codebook_loss: mx.array
    commitment_loss: mx.array
    semantic_distill_z: Optional[mx.array] = None


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.dwconv = CausalConvNet(
            dim, dim, kernel_size=kernel_size, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            mx.ones((dim,)) * layer_scale_init_value
            if layer_scale_init_value > 0
            else None
        )

    def __call__(self, x: mx.array, apply_residual: bool = True) -> mx.array:
        inp = x
        x = self.dwconv(x)
        x = x.swapaxes(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.swapaxes(1, 2)
        return inp + x if apply_residual else x


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: mx.Dtype = mx.float32
) -> mx.array:
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2, dtype=mx.float32) / n_elem))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1).astype(dtype)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    freqs = freqs_cis.reshape(1, xshaped.shape[1], 1, xshaped.shape[3], 2)
    x_out = mx.stack(
        [
            xshaped[..., 0] * freqs[..., 0] - xshaped[..., 1] * freqs[..., 1],
            xshaped[..., 1] * freqs[..., 0] + xshaped[..., 0] * freqs[..., 1],
        ],
        axis=-1,
    )
    return x_out.reshape(x.shape).astype(x.dtype)


class TFRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        out = x.astype(mx.float32)
        out = out * mx.rsqrt(mx.mean(out * out, axis=-1, keepdims=True) + self.eps)
        return out.astype(x.dtype) * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_values

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True
    pos_embed_type: str = "rope"
    max_relative_position: int = 128

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.pos_embed_type = config.pos_embed_type

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array | None,
        mask: mx.array | None,
    ) -> mx.array:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        q = qkv[..., :kv_size]
        k = qkv[..., kv_size : 2 * kv_size]
        v = qkv[..., 2 * kv_size : 3 * kv_size]
        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim)
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope" and freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if self.n_local_heads != self.n_head:
            repeat_factor = self.n_head // self.n_local_heads
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)

        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, self.head_dim * self.n_head)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = TFRMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = TFRMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, init_values=1e-2)
        self.ffn_layer_scale = LayerScale(config.dim, init_values=1e-2)

    def __call__(
        self, x: mx.array, freqs_cis: mx.array | None, mask: mx.array
    ) -> mx.array:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask)
        )
        return h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = TFRMSNorm(config.dim, eps=config.norm_eps)
        self.freqs_cis = (
            precompute_freqs_cis(
                config.block_size,
                config.head_dim,
                int(config.rope_base),
                dtype=mx.float32,
            )
            if config.pos_embed_type == "rope"
            else None
        )
        row = mx.arange(config.block_size)[:, None]
        col = mx.arange(config.block_size)[None, :]
        self.causal_mask = row >= col

    def __call__(
        self, x: mx.array, input_pos: mx.array, mask: mx.array | None = None
    ) -> mx.array:
        freqs_cis = self.freqs_cis[input_pos] if self.freqs_cis is not None else None
        if mask is None:
            local_mask = self.causal_mask[input_pos][:, input_pos]
            mask = local_mask[None, None, :, :]
        if mask.dtype == mx.bool_:
            mask = mx.where(mask, 0.0, -1e9).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        return self.norm(x)


class WindowLimitedTransformer(Transformer):
    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        causal: bool = True,
        look_ahead_conv: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = (
            look_ahead_conv if look_ahead_conv is not None else Identity()
        )
        self.input_proj = (
            nn.Linear(input_dim, config.dim) if input_dim != config.dim else Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim) if input_dim != config.dim else Identity()
        )

    def make_window_limited_mask(self, max_length: int) -> mx.array:
        row_indices = mx.arange(max_length)[:, None]
        col_indices = mx.arange(max_length)[None, :]
        window_size = self.window_size or max_length
        valid_range = mx.maximum(row_indices - window_size + 1, 0)
        mask = (col_indices >= valid_range) & (col_indices <= row_indices)
        return mask[None, None, :, :]

    def make_mask(self, max_length: int) -> mx.array:
        row_indices = mx.arange(max_length)[:, None]
        col_indices = mx.arange(max_length)[None, :]
        return (col_indices <= row_indices)[None, None, :, :]

    def __call__(self, x: mx.array, x_lens: Optional[mx.array] = None) -> mx.array:
        if self.channels_first:
            x = x.swapaxes(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        input_pos = mx.arange(x.shape[1], dtype=mx.int32)
        max_length = int(x.shape[1])
        mask = (
            self.make_window_limited_mask(max_length)
            if self.window_size is not None
            else self.make_mask(max_length)
        )
        x = super().__call__(x, input_pos, mask)
        x = self.output_proj(x)
        return x.swapaxes(1, 2) if self.channels_first else x


class DownsampleResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int, ...] = (2, 2),
        downsample_dims: Optional[tuple[int, ...]] = None,
        pre_module: Optional[nn.Module] = None,
        post_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        if downsample_dims is None:
            downsample_dims = tuple(input_dim for _ in range(len(downsample_factor)))

        all_dims = (input_dim,) + tuple(downsample_dims)
        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )
        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.downsample = [
            [
                CausalConvNet(
                    all_dims[idx],
                    all_dims[idx + 1],
                    kernel_size=factor,
                    stride=factor,
                ),
                ConvNeXtBlock(dim=all_dims[idx + 1]),
            ]
            for idx, factor in enumerate(downsample_factor)
        ]
        self.upsample = [
            [
                CausalTransConvNet(
                    all_dims[idx + 1],
                    all_dims[idx],
                    kernel_size=factor,
                    stride=factor,
                ),
                ConvNeXtBlock(dim=all_dims[idx]),
            ]
            for idx, factor in reversed(list(enumerate(downsample_factor)))
        ]
        self.pre_module = pre_module if pre_module is not None else Identity()
        self.post_module = post_module if post_module is not None else Identity()

    def __call__(
        self,
        z: mx.array,
        n_quantizers: Optional[int] = None,
        semantic_len: Optional[mx.array] = None,
    ) -> VQResult:
        del semantic_len
        original_shape = z.shape
        for block in self.downsample:
            for layer in block:
                z = layer(z)
        z = self.pre_module(z)

        (
            semantic_z,
            semantic_codes,
            semantic_latents,
            semantic_commitment_loss,
            semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = mx.concatenate([semantic_codes, codes], axis=1)
        latents = mx.concatenate([semantic_latents, latents], axis=1)

        z = self.post_module(z)
        for block in self.upsample:
            for layer in block:
                z = layer(z)
        diff = int(original_shape[-1] - z.shape[-1])
        if diff > 0:
            z = mx.pad(z, [(0, 0), (0, 0), (diff, 0)])
        elif diff < 0:
            z = z[..., abs(diff) :]

        return VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def decode(self, indices: mx.array) -> mx.array:
        _validate_codebook_indices(
            indices[:, :1],
            self.semantic_quantizer.codebook_size,
            "semantic code IDs out of range",
        )
        if indices.shape[1] > 1:
            _validate_codebook_indices(
                indices[:, 1:],
                self.quantizer.codebook_size,
                "residual code IDs out of range",
            )

        z_q_sem = self.semantic_quantizer.from_codes(indices[:, :1])[0]
        z_q_res = (
            self.quantizer.from_codes(indices[:, 1:])[0]
            if indices.shape[1] > 1
            else mx.zeros_like(z_q_sem)
        )
        z_q = self.post_module(z_q_sem + z_q_res)
        for block in self.upsample:
            for layer in block:
                z_q = layer(z_q)
        return z_q


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = [
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        ]
        self.causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.block:
            y = layer(y)
        pad = int(x.shape[-1] - y.shape[-1])
        if pad > 0:
            x = x[..., :-pad] if self.causal else x[..., pad // 2 : -pad // 2]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            Identity()
            if n_t_layer == 0
            else WindowLimitedTransformer(
                causal=causal,
                input_dim=dim,
                window_size=512,
                config=transformer_general_config(
                    n_layer=n_t_layer,
                    n_head=dim // 64,
                    dim=dim,
                    intermediate_size=dim * 3,
                ),
            )
        )
        self.block = [
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            transformer_module,
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list[int] = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list[int] = [0, 0, 4, 4],
        transformer_general_config: Optional[ModelArgs] = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers: list[nn.Module] = [conv_class(1, d_model, kernel_size=7, padding=3)]
        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            layers.append(
                EncoderBlock(
                    d_model,
                    stride=stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            )
        layers += [
            Snake1d(d_model),
            conv_class(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = layers
        self.enc_dim = d_model

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        self.block = [
            Snake1d(input_dim),
            conv_trans_class(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        d_out: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers: list[nn.Module] = [
            conv_class(input_channel, channels, kernel_size=7, padding=3)
        ]
        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride, causal=causal))
        layers += [
            Snake1d(output_dim),
            conv_class(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.model:
            x = layer(x)
        return x


class DAC(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 8, 4, 2],
        quantizer: Optional[nn.Module] = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: list[int] = [0, 0, 0, 0],
        decoder_transformer_layers: list[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        del decoder_transformer_layers
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = int(np.prod(encoder_rates))
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.quantizer = quantizer
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
        )
        self.frame_length = self.hop_length * 4

    def decode(self, indices: mx.array, feature_lengths: mx.array):
        if indices.ndim == 2:
            indices = indices[None]
        z = self.quantizer.decode(indices)
        audio_lengths = feature_lengths * self.frame_length
        return self.decoder(z), audio_lengths

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        wn_prefixes = set()
        for key in weights:
            marker = ".conv.parametrizations.weight.original0"
            if marker in key:
                wn_prefixes.add(key.split(marker)[0])

        out: dict[str, mx.array] = {}
        for key, value in weights.items():
            if ".conv.parametrizations.weight.original0" in key:
                key = key.replace(
                    ".conv.parametrizations.weight.original0", ".weight_g"
                )
            elif ".conv.parametrizations.weight.original1" in key:
                key = key.replace(
                    ".conv.parametrizations.weight.original1", ".weight_v"
                )
            elif key.endswith(".conv.bias"):
                prefix = key[: -len(".conv.bias")]
                if prefix in wn_prefixes:
                    key = prefix + ".bias"
            elif ".parametrizations.weight.original0" in key:
                key = key.replace(".parametrizations.weight.original0", ".weight_g")
            elif ".parametrizations.weight.original1" in key:
                key = key.replace(".parametrizations.weight.original1", ".weight_v")
            out[key] = value
        return out


def build_ae(**cfg) -> DAC:
    q_config = ModelArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=cfg.get("latent_dim", 1024),
        intermediate_size=3072,
        head_dim=64,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    latent_dim = cfg.get("latent_dim", 1024)

    def make_transformer():
        return WindowLimitedTransformer(
            causal=True,
            window_size=128,
            input_dim=latent_dim,
            config=q_config,
        )

    quantizer = DownsampleResidualVectorQuantize(
        input_dim=latent_dim,
        n_codebooks=cfg.get("n_codebooks", 9),
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        semantic_codebook_size=cfg.get("semantic_codebook_size", 4096),
        pre_module=make_transformer(),
        post_module=make_transformer(),
    )

    def transformer_general_config(**kw):
        return ModelArgs(
            block_size=kw.get("block_size", 16384),
            n_layer=kw.get("n_layer", 8),
            n_head=kw.get("n_head", 8),
            dim=kw.get("dim", 512),
            intermediate_size=kw.get("intermediate_size", 1536),
            n_local_heads=kw.get("n_local_heads", -1),
            head_dim=kw.get("head_dim", 64),
            rope_base=kw.get("rope_base", 10000),
            norm_eps=kw.get("norm_eps", 1e-5),
            dropout_rate=kw.get("dropout_rate", 0.1),
            attn_dropout_rate=kw.get("attn_dropout_rate", 0.1),
            channels_first=kw.get("channels_first", True),
        )

    return DAC(
        encoder_dim=cfg.get("encoder_dim", 64),
        encoder_rates=[2, 4, 8, 8],
        latent_dim=latent_dim,
        decoder_dim=cfg.get("decoder_dim", 1536),
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=cfg.get("sample_rate", 44100),
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=transformer_general_config,
    )


@dataclass
class FishCodecModel:
    model: DAC
    config: FishCodecConfig
    sample_rate: int

    @classmethod
    def from_dir(cls, codec_dir: str | Path) -> "FishCodecModel":
        resolved = validate_codec_assets(codec_dir)
        config = FishCodecConfig(
            **json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        )
        model = build_ae(**config.to_dict())
        weights = model.sanitize(mx.load(str(resolved / "model.safetensors")))
        weights = _add_causal_conv_alias_weights(model, weights)
        _validate_codec_weights(model, weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        return cls(model=model, config=config, sample_rate=config.sample_rate)

    def encode(self, waveform: mx.array) -> mx.array:
        """Encode a mono waveform to VQ codes.

        Args:
            waveform: (T,) float32 mono audio at self.sample_rate.

        Returns:
            codes: (n_codebooks + 1, T_frames) int32 codebook indices.
        """
        audio = waveform[None, None, :]
        length = int(audio.shape[-1])
        pad_amount = math.ceil(length / self.model.frame_length) * self.model.frame_length - length
        if pad_amount > 0:
            audio = mx.pad(audio, [(0, 0), (0, 0), (0, pad_amount)])
        z = self.model.encoder(audio)
        vq_result = self.model.quantizer(z)
        return vq_result.codes[0]

    def decode(self, codes: mx.array) -> mx.array:
        expected_rows = self.config.n_codebooks + 1
        if codes.ndim != 2 or codes.shape[0] != expected_rows:
            raise ValueError(f"expected {expected_rows} Fish codebook rows for decode")
        _validate_codebook_indices(
            codes[:1],
            self.model.quantizer.semantic_quantizer.codebook_size,
            "semantic code IDs out of range",
        )
        batch_codes = codes[None, :, :]
        if codes.shape[0] > 1:
            residual_codes = batch_codes[:, 1:, :]
            _validate_nonnegative_codebook_indices(
                residual_codes,
                "residual code IDs out of range",
            )
            batch_codes[:, 1:, :] = mx.minimum(
                residual_codes,
                self.model.quantizer.quantizer.codebook_size - 1,
            )
        feature_lengths = mx.array([codes.shape[-1]], dtype=mx.int32)
        waveform, audio_lengths = self.model.decode(batch_codes, feature_lengths)
        waveform = waveform.squeeze(0)
        target_length = int(audio_lengths[0])
        return waveform[..., :target_length]


def _validate_codec_weights(model: DAC, weights: dict[str, mx.array]) -> None:
    model_params = tree_flatten(model.parameters(), destination={})
    missing = sorted(set(model_params) - set(weights))
    unexpected = sorted(set(weights) - set(model_params))
    mismatched = sorted(
        key
        for key in set(model_params) & set(weights)
        if model_params[key].shape != weights[key].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:3]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:3]}")
        raise ValueError(
            "codec weights do not match model parameters"
            + (f": {', '.join(details)}" if details else "")
        )


def _add_causal_conv_alias_weights(
    model: DAC, weights: dict[str, mx.array]
) -> dict[str, mx.array]:
    model_params = tree_flatten(model.parameters(), destination={})
    augmented = dict(weights)

    for key, param in model_params.items():
        if key in augmented:
            continue
        if key.endswith(".conv.conv.weight"):
            source_key = key[: -len(".conv.conv.weight")] + ".weight_v"
        elif key.endswith(".conv.conv.bias"):
            source_key = key[: -len(".conv.conv.bias")] + ".bias"
        else:
            continue

        source = augmented.get(source_key)
        if source is None or source.shape != param.shape:
            continue
        augmented[key] = source

    return augmented


def _validate_codebook_indices(
    indices: mx.array, codebook_size: int, message: str
) -> None:
    if indices.size == 0:
        return
    min_id = int(mx.min(indices).item())
    max_id = int(mx.max(indices).item())
    if min_id < 0 or max_id >= codebook_size:
        raise ValueError(message)


def _validate_nonnegative_codebook_indices(indices: mx.array, message: str) -> None:
    if indices.size == 0:
        return
    min_id = int(mx.min(indices).item())
    if min_id < 0:
        raise ValueError(message)
