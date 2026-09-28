"""Qwen3-TTS 12.5 Hz reference encoder.

The graph is the causal Mimi encoder used by the bundled audio tokenizer:
SEANet, a sliding-window transformer, a stride-2 downsample, then the first
16 of the split residual quantizers. Convolution weights are already in MLX
layout.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

DOWNSAMPLE_RATE = 1920
VALID_QUANTIZERS = 16
_SLIDING_WINDOW = 250


def _conv_padding(
    length: int, kernel: int, stride: int, dilation: int, *, causal: bool
) -> tuple[int, int]:
    kernel_size = (kernel - 1) * dilation + 1
    padding_total = kernel_size - stride
    frames = math.ceil((length - kernel_size + padding_total) / stride + 1) - 1
    extra = frames * stride + kernel_size - padding_total - length
    if causal:
        return padding_total, extra
    right = padding_total // 2
    return padding_total - right, right + extra


class _Elu(nn.Module):
    def __call__(self, hidden: mx.array) -> mx.array:
        return nn.elu(hidden)


class _MimiConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        replicate: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.replicate = replicate
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        left, right = _conv_padding(
            int(hidden.shape[1]),
            self.kernel,
            self.stride,
            self.dilation,
            causal=True,
        )
        if left or right:
            mode = "edge" if self.replicate else "constant"
            hidden = mx.pad(hidden, [(0, 0), (left, right), (0, 0)], mode=mode)
        return self.conv(hidden)


class _ResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.block = [
            _Elu(),
            _MimiConv(channels, hidden, 3, dilation=1),
            _Elu(),
            _MimiConv(hidden, channels, 1),
        ]

    def __call__(self, hidden: mx.array) -> mx.array:
        residual = hidden
        for layer in self.block:
            hidden = layer(hidden)
        return residual + hidden


class _SeanetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 64
        layers: list[nn.Module] = [_MimiConv(1, channels, 7)]
        for stride in (4, 5, 6, 8):
            layers.append(_ResidualUnit(channels))
            layers.append(_Elu())
            layers.append(_MimiConv(channels, channels * 2, stride * 2, stride=stride))
            channels *= 2
        layers.extend([_Elu(), _MimiConv(channels, 512, 3)])
        self.layers = layers

    def __call__(self, hidden: mx.array) -> mx.array:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _sliding_causal_mask(length: int, dtype: mx.Dtype) -> mx.array:
    query = mx.arange(length)[:, None]
    key = mx.arange(length)[None, :]
    blocked = (key > query) | (key <= query - _SLIDING_WINDOW)
    minimum = mx.array(mx.finfo(dtype).min, dtype=dtype)
    return mx.where(blocked, minimum, mx.array(0, dtype=dtype))[None, None, :, :]


def _rotate_half(hidden: mx.array) -> mx.array:
    half = hidden.shape[-1] // 2
    return mx.concatenate([-hidden[..., half:], hidden[..., :half]], axis=-1)


class _Rope:
    def __init__(self) -> None:
        self.inv_freq = tuple(
            1.0 / (10_000.0 ** (index / 64.0)) for index in range(0, 64, 2)
        )

    def __call__(
        self, position_ids: mx.array, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        freqs = position_ids.astype(mx.float32)[..., None] * mx.array(
            self.inv_freq, dtype=mx.float32
        )
        angles = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(512, 512, bias=False)
        self.k_proj = nn.Linear(512, 512, bias=False)
        self.v_proj = nn.Linear(512, 512, bias=False)
        self.o_proj = nn.Linear(512, 512, bias=False)
        self.rope = _Rope()

    def __call__(self, hidden: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
        key = self.k_proj(hidden).reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
        value = self.v_proj(hidden).reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
        cos, sin = self.rope(mx.arange(length)[None, :], hidden.dtype)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(64.0),
            mask=_sliding_causal_mask(length, hidden.dtype),
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, 512))


class _LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = mx.full((512,), 0.01)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.scale * hidden


class _TransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(512, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(512, eps=1e-5)
        self.self_attn = _Attention()
        self.self_attn_layer_scale = _LayerScale()
        self.mlp_layer_scale = _LayerScale()
        self.mlp = _Mlp()

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden + self.self_attn_layer_scale(
            self.self_attn(self.input_layernorm(hidden))
        )
        return hidden + self.mlp_layer_scale(
            self.mlp(self.post_attention_layernorm(hidden))
        )


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(512, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 512, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(hidden)))


class _EncoderCodebook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cluster_usage = mx.ones((2048,))
        self.embed_sum = mx.zeros((2048, 256))
        self.initialized = mx.ones((1,))

    def embed(self) -> mx.array:
        usage = mx.maximum(
            self.cluster_usage, mx.array(1e-5, dtype=self.cluster_usage.dtype)
        )
        return self.embed_sum / usage[:, None]

    def encode(self, hidden: mx.array) -> mx.array:
        flat = hidden.reshape(-1, hidden.shape[-1]).astype(mx.float32)
        centroids = self.embed().astype(mx.float32)
        distance = (
            mx.sum(flat * flat, axis=-1, keepdims=True)
            + mx.sum(centroids * centroids, axis=-1)
            - 2.0 * flat @ centroids.T
        )
        return mx.argmin(distance, axis=-1).reshape(hidden.shape[:-1])

    def decode(self, codes: mx.array) -> mx.array:
        return self.embed()[codes]


class _VectorQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codebook = _EncoderCodebook()

    def encode(self, hidden: mx.array) -> mx.array:
        return self.codebook.encode(hidden)

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes)


class _ResidualQuantizer(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(512, 256, 1, bias=False)
        self.output_proj = nn.Conv1d(256, 512, 1, bias=False)
        self.layers = [_VectorQuantizer() for _ in range(count)]

    def encode(self, hidden: mx.array, count: int) -> mx.array:
        residual = self.input_proj(hidden)
        indices = []
        for layer in self.layers[:count]:
            codes = layer.encode(residual)
            residual = residual - layer.decode(codes)
            indices.append(codes)
            mx.eval(residual, codes)
        return mx.stack(indices, axis=1)


class CodecEncoder(nn.Module):
    """Weight-key root: parameters begin with ``encoder.``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _SeanetEncoder()
        self.encoder_transformer = _TransformerStack()
        self.downsample = _MimiConv(512, 512, 4, stride=2, bias=False, replicate=True)
        self.quantizer = _SplitQuantizer()

    def encode(self, samples: mx.array) -> mx.array:
        """Return reference codes ``(frames, 16)`` for one mono waveform."""

        if samples.ndim != 1:
            raise ValueError(f"Expected a mono waveform, got {tuple(samples.shape)}.")
        if samples.shape[0] <= 0:
            raise ValueError("Reference audio is empty.")
        hidden = self.encoder(samples.astype(mx.float32)[None, :, None])
        hidden = self.encoder_transformer(hidden)
        hidden = self.downsample(hidden)
        codes = self.quantizer.encode(hidden)
        keep = math.ceil(int(samples.shape[0]) / DOWNSAMPLE_RATE)
        if int(codes.shape[-1]) < keep:
            raise ValueError(
                f"Encoder produced {int(codes.shape[-1])} frames for {keep} valid frames."
            )
        trimmed = mx.swapaxes(codes[0, :, :keep], 0, 1).astype(mx.int32)
        if int(trimmed.shape[1]) != VALID_QUANTIZERS:
            raise ValueError(
                f"Expected {(keep, VALID_QUANTIZERS)} reference codes, got {tuple(trimmed.shape)}."
            )
        mx.eval(trimmed)
        return trimmed


class _TransformerStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_TransformerLayer() for _ in range(8)]

    def __call__(self, hidden: mx.array) -> mx.array:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _SplitQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_residual_vector_quantizer = _ResidualQuantizer(1)
        self.acoustic_residual_vector_quantizer = _ResidualQuantizer(31)

    def encode(self, hidden: mx.array) -> mx.array:
        semantic = self.semantic_residual_vector_quantizer.encode(hidden, 1)
        acoustic = self.acoustic_residual_vector_quantizer.encode(
            hidden, VALID_QUANTIZERS - 1
        )
        return mx.concatenate([semantic, acoustic], axis=1)
