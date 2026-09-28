"""Qwen3-TTS 12 Hz waveform decoder.

Convolutions stay in MLX layout, channels last. Codebooks are recovered from
``embedding_sum / cluster_usage``; the checkpoint does not store a lookup table.
Every transformer layer is sliding attention over the last 72 frames, as in
the official decoder config and its streaming cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .audio_encoder import CodecEncoder

_SLIDING_WINDOW = 72


def _causal_mask(
    query_length: int, key_length: int, dtype: mx.Dtype
) -> mx.array | None:
    """Queries sit at the end of the keys and see at most the last window."""

    if query_length == 1 and key_length <= _SLIDING_WINDOW:
        return None
    query_positions = mx.arange(key_length - query_length, key_length)[:, None]
    key_positions = mx.arange(key_length)[None, :]
    blocked = (key_positions > query_positions) | (
        key_positions <= query_positions - _SLIDING_WINDOW
    )
    minimum = mx.array(mx.finfo(dtype).min, dtype=dtype)
    return mx.where(blocked, minimum, mx.array(0, dtype=dtype))[None, None, :, :]


@dataclass
class _WindowKV:
    """One layer's keys and values for the frames a later query can still see."""

    keys: mx.array | None = None
    values: mx.array | None = None

    def extend(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is not None and self.values is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys = keys[:, :, -(_SLIDING_WINDOW - 1) :, :]
        self.values = values[:, :, -(_SLIDING_WINDOW - 1) :, :]
        return keys, values


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        return self.conv(x)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + mx.square(mx.sin(x * alpha)) / (beta + 1e-9)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,)) * 1e-6

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        hidden = self.dwconv(x)
        hidden = self.pwconv2(nn.gelu(self.pwconv1(self.norm(hidden))))
        return residual + self.gamma * hidden


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.cluster_usage = mx.ones((codebook_size,))
        self.embedding_sum = mx.zeros((codebook_size, dim))

    def decode(self, codes: mx.array) -> mx.array:
        usage = mx.maximum(self.cluster_usage, mx.array(self.epsilon))
        embedding = self.embedding_sum / usage[:, None]
        return embedding[codes]


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = EuclideanCodebook(dim, codebook_size)

    def decode(self, codes: mx.array) -> mx.array:
        return mx.transpose(self.codebook.decode(codes), (0, 2, 1))


class ResidualQuantizers(nn.Module):
    def __init__(self, count: int, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.layers = [VectorQuantizer(dim, codebook_size) for _ in range(count)]

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.layers[0].decode(codes[0])
        for index, layer in enumerate(self.layers[1:], start=1):
            quantized = quantized + layer.decode(codes[index])
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(512, 256, 1, bias=False)
        self.output_proj = nn.Conv1d(256, 512, 1, bias=False)
        self.vq = ResidualQuantizers(count, 256, 2048)

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.vq.decode(mx.transpose(codes, (1, 0, 2)))
        quantized = mx.transpose(quantized, (0, 2, 1))
        return mx.transpose(self.output_proj(quantized), (0, 2, 1))


class SplitQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rvq_first = ResidualVectorQuantizer(1)
        self.rvq_rest = ResidualVectorQuantizer(15)

    def decode(self, codes: mx.array) -> mx.array:
        return self.rvq_first.decode(codes[:, :1]) + self.rvq_rest.decode(codes[:, 1:])


class DecoderAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 16
        self.head_dim = 64
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(512, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(512, self.heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(512, self.heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, 512, bias=False)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None,
        cache: _WindowKV | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        query = (
            self.q_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        key = (
            self.k_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        value = (
            self.v_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        cos, sin = rope
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        if cache is not None:
            key, value = cache.extend(key, value)
            mask = _causal_mask(length, int(key.shape[2]), x.dtype)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = 1e-5

    def __call__(self, x: mx.array) -> mx.array:
        centered = x.astype(mx.float32)
        variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
        return (self.weight * centered * mx.rsqrt(variance + self.eps)).astype(x.dtype)


class _Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = mx.ones((512,)) * 0.01

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class TransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = DecoderAttention()
        self.input_layernorm = _RMSNorm(512)
        self.post_attention_layernorm = _RMSNorm(512)
        self.self_attn_layer_scale = _Scale()
        self.mlp_layer_scale = _Scale()
        self.mlp = _MLP()

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None,
        cache: _WindowKV | None = None,
    ) -> mx.array:
        x = x + self.self_attn_layer_scale(
            self.self_attn(self.input_layernorm(x), rope, mask, cache)
        )
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(512, 1024, bias=False)
        self.up_proj = nn.Linear(512, 1024, bias=False)
        self.down_proj = nn.Linear(1024, 512, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class PreTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1024, 512)
        self.output_proj = nn.Linear(512, 1024)
        self.layers = [TransformerLayer() for _ in range(8)]
        self.norm = _RMSNorm(512)
        self._inv_freq = tuple(
            1.0 / (10_000.0 ** (index / 64.0)) for index in range(0, 64, 2)
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.input_proj(x)
        length = hidden.shape[1]
        positions = mx.arange(length)[None, :]
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        freqs = positions.astype(mx.float32)[..., None] * inv_freq
        angles = mx.concatenate([freqs, freqs], axis=-1)
        rope = (
            mx.cos(angles).astype(hidden.dtype),
            mx.sin(angles).astype(hidden.dtype),
        )
        mask = _causal_mask(length, length, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask)
        return self.output_proj(self.norm(hidden))

    def step(self, x: mx.array, caches: list[_WindowKV], position: int) -> mx.array:
        hidden = self.input_proj(x)
        length = int(hidden.shape[1])
        positions = mx.arange(position, position + length)[None, :]
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        freqs = positions.astype(mx.float32)[..., None] * inv_freq
        angles = mx.concatenate([freqs, freqs], axis=-1)
        rope = (
            mx.cos(angles).astype(hidden.dtype),
            mx.sin(angles).astype(hidden.dtype),
        )
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden = layer(hidden, rope, None, cache)
        return self.output_proj(self.norm(hidden))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, 7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv2(self.act2(self.conv1(self.act1(x))))
        return x + hidden


class UpsampleConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rate: int) -> None:
        super().__init__()
        kernel = 2 * rate
        self.trim = kernel - rate
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel, stride=rate, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv(x)
        if self.trim:
            hidden = hidden[:, : -self.trim, :]
        return hidden


class DecoderBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        rates = (8, 5, 4, 3)
        in_dim = 1536 // 2**index
        out_dim = 1536 // 2 ** (index + 1)
        self.block = [
            SnakeBeta(in_dim),
            UpsampleConv(in_dim, out_dim, rates[index]),
            ResidualUnit(out_dim, 1),
            ResidualUnit(out_dim, 3),
            ResidualUnit(out_dim, 9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class _Transpose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel, stride=kernel, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class CodecDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_conv = CausalConv1d(512, 1024, 3)
        self.pre_transformer = PreTransformer()
        self.quantizer = SplitQuantizer()
        self.upsample = [
            [_Transpose(1024, 1024, 2), ConvNeXtBlock(1024)],
            [_Transpose(1024, 1024, 2), ConvNeXtBlock(1024)],
        ]
        self.decoder = [
            CausalConv1d(1024, 1536, 7),
            DecoderBlock(0),
            DecoderBlock(1),
            DecoderBlock(2),
            DecoderBlock(3),
            SnakeBeta(1536 // 16),
            CausalConv1d(1536 // 16, 1, 7),
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        if codes.ndim != 3 or int(codes.shape[1]) != 16:
            raise ValueError(
                f"Expected codes (batch, 16, frames), got {tuple(codes.shape)}."
            )
        hidden = mx.transpose(self.quantizer.decode(codes), (0, 2, 1))
        hidden = self.pre_conv(hidden)
        hidden = self.pre_transformer(hidden)
        for transpose, block in self.upsample:
            hidden = block(transpose(hidden))
        for layer in self.decoder:
            hidden = layer(hidden)
        return mx.clip(mx.transpose(hidden, (0, 2, 1)), -1.0, 1.0)


def _causal_conv_step(
    conv: CausalConv1d, hidden: mx.array, buffer: mx.array | None
) -> tuple[mx.array, mx.array | None]:
    if conv.padding == 0:
        return conv.conv(hidden), buffer
    if buffer is None:
        buffer = mx.zeros(
            (hidden.shape[0], conv.padding, hidden.shape[-1]), dtype=hidden.dtype
        )
    window = mx.concatenate([buffer, hidden], axis=1)
    return conv.conv(window), window[:, -conv.padding :, :]


def _upsample_step(
    layer: UpsampleConv, hidden: mx.array, tail: mx.array | None
) -> tuple[mx.array, mx.array | None]:
    """Overlap-add transposed-convolution tails without counting bias twice."""

    raw = layer.conv(hidden)
    bias = layer.conv.bias
    if bias is not None:
        raw = raw - bias
    if layer.trim == 0:
        return raw if bias is None else raw + bias, None
    if tail is not None:
        raw = mx.concatenate(
            [raw[:, : layer.trim, :] + tail, raw[:, layer.trim :, :]], axis=1
        )
    emitted = raw[:, : -layer.trim, :]
    if bias is not None:
        emitted = emitted + bias
    return emitted, raw[:, -layer.trim :, :]


def _convnext_step(
    block: ConvNeXtBlock, hidden: mx.array, buffer: mx.array | None
) -> tuple[mx.array, mx.array | None]:
    residual = hidden
    hidden, buffer = _causal_conv_step(block.dwconv, hidden, buffer)
    hidden = block.pwconv2(nn.gelu(block.pwconv1(block.norm(hidden))))
    return residual + block.gamma * hidden, buffer


def _residual_step(
    unit: ResidualUnit,
    hidden: mx.array,
    buffers: tuple[mx.array | None, mx.array | None],
) -> tuple[mx.array, tuple[mx.array | None, mx.array | None]]:
    first, second = buffers
    hidden_conv, first = _causal_conv_step(unit.conv1, unit.act1(hidden), first)
    hidden_conv, second = _causal_conv_step(unit.conv2, unit.act2(hidden_conv), second)
    return hidden + hidden_conv, (first, second)


@dataclass
class CodecDecodeStream:
    """Incremental decoder state for one utterance. It is not kept on the module."""

    decoder: CodecDecoder
    conv_buffers: dict[int, mx.array | None] = field(default_factory=dict)
    residual_buffers: dict[int, tuple[mx.array | None, mx.array | None]] = field(
        default_factory=dict
    )
    tails: dict[int, mx.array | None] = field(default_factory=dict)
    caches: list[_WindowKV] | None = None
    position: int = 0

    def push(self, codes: mx.array) -> mx.array:
        """Decode new codes of shape ``(batch, 16, frames)`` and return finalized audio."""

        if codes.ndim != 3 or int(codes.shape[1]) != 16 or int(codes.shape[2]) <= 0:
            raise ValueError(
                f"Expected new codes (batch, 16, frames), got {tuple(codes.shape)}."
            )
        decoder = self.decoder
        hidden = mx.transpose(decoder.quantizer.decode(codes), (0, 2, 1))
        hidden, buffer = _causal_conv_step(
            decoder.pre_conv, hidden, self.conv_buffers.get(id(decoder.pre_conv))
        )
        self.conv_buffers[id(decoder.pre_conv)] = buffer
        if self.caches is None:
            self.caches = [_WindowKV() for _ in decoder.pre_transformer.layers]
        hidden = decoder.pre_transformer.step(hidden, self.caches, self.position)
        self.position += int(hidden.shape[1])
        for transpose, block in decoder.upsample:
            hidden = transpose(hidden)
            hidden, buffer = _convnext_step(
                block, hidden, self.conv_buffers.get(id(block))
            )
            self.conv_buffers[id(block)] = buffer
        for layer in decoder.decoder:
            hidden = self._layer(layer, hidden)
        audio = mx.clip(mx.transpose(hidden, (0, 2, 1)), -1.0, 1.0)
        self._realize(audio)
        return audio

    def _layer(self, layer: nn.Module, hidden: mx.array) -> mx.array:
        if isinstance(layer, CausalConv1d):
            hidden, buffer = _causal_conv_step(
                layer, hidden, self.conv_buffers.get(id(layer))
            )
            self.conv_buffers[id(layer)] = buffer
            return hidden
        if isinstance(layer, SnakeBeta):
            return layer(hidden)
        if not isinstance(layer, DecoderBlock):
            raise TypeError(f"Unsupported decoder layer {type(layer).__name__}.")
        for part in layer.block:
            if isinstance(part, SnakeBeta):
                hidden = part(hidden)
            elif isinstance(part, UpsampleConv):
                hidden, tail = _upsample_step(part, hidden, self.tails.get(id(part)))
                self.tails[id(part)] = tail
            elif isinstance(part, ResidualUnit):
                hidden, buffers = _residual_step(
                    part, hidden, self.residual_buffers.get(id(part), (None, None))
                )
                self.residual_buffers[id(part)] = buffers
            else:
                raise TypeError(f"Unsupported decoder block {type(part).__name__}.")
        return hidden

    def _realize(self, audio: mx.array) -> None:
        pending = [audio]
        pending.extend(
            value for value in self.conv_buffers.values() if value is not None
        )
        pending.extend(value for value in self.tails.values() if value is not None)
        for buffers in self.residual_buffers.values():
            pending.extend(value for value in buffers if value is not None)
        for cache in self.caches or []:
            pending.extend(
                value for value in (cache.keys, cache.values) if value is not None
            )
        mx.eval(*pending)

    def close(self) -> None:
        """Drop this utterance's buffers. A later request allocates its own."""

        self.conv_buffers.clear()
        self.residual_buffers.clear()
        self.tails.clear()
        self.caches = None
        self.position = 0


def codec_parameter_key(checkpoint_key: str) -> str:
    """MLX does not register modules whose names start with an underscore."""

    return checkpoint_key.replace("._codebook.", ".codebook.")


class SpeechCodec(nn.Module):
    """Decoder and reference encoder for the bundled audio tokenizer."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = CodecDecoder()
        self.encoder = CodecEncoder()

    def decode(self, codes: mx.array) -> mx.array:
        return self.decoder(codes)

    def encode(self, samples: mx.array) -> mx.array:
        return self.encoder.encode(samples)
