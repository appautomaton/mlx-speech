"""Small MLX-native layers shared by dots.tts continuous components."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .._cache import BoundedKVCache


class _FusedLinear(nn.Module):
    """One weight-exact projection assembled from adjacent Linear layers."""

    def __init__(self, layers: tuple[nn.Linear, ...]):
        super().__init__()
        if not layers:
            raise ValueError("fused linear requires at least one source layer")
        input_dims = int(layers[0].weight.shape[1])
        weight_dtype = layers[0].weight.dtype
        has_bias = "bias" in layers[0]
        if any(
            int(layer.weight.shape[1]) != input_dims
            or layer.weight.dtype != weight_dtype
            or ("bias" in layer) != has_bias
            for layer in layers
        ):
            raise ValueError("fused linear source projections are incompatible")
        self.weight = mx.concatenate(tuple(layer.weight for layer in layers), axis=0)
        if has_bias:
            self.bias = mx.concatenate(tuple(layer.bias for layer in layers), axis=0)
        self.freeze()

    def __call__(self, value: mx.array) -> mx.array:
        if "bias" in self:
            return mx.addmm(self["bias"], value, self["weight"].T)
        return value @ self["weight"].T


class CausalConv1d(nn.Module):
    """Channels-last causal convolution with explicit MLX-layout weights."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if min(input_channels, output_channels, kernel_size, stride, dilation) <= 0:
            raise ValueError("causal convolution dimensions must be positive")
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.left_padding = self.dilation * (int(kernel_size) - 1)
        scale = math.sqrt(2.0 / (input_channels * kernel_size + output_channels))
        self.weight = (
            mx.random.normal((output_channels, kernel_size, input_channels)) * scale
        )
        self.bias = mx.zeros((output_channels,)) if bias else None

    def _convolve(self, value: mx.array) -> mx.array:
        output = mx.conv1d(
            value,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )
        return output if self.bias is None else output + self.bias

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 3 or int(value.shape[-1]) != int(self.weight.shape[-1]):
            raise ValueError(
                "causal convolution expects (batch, time, input_channels), "
                f"got {value.shape}"
            )
        if self.left_padding:
            value = mx.pad(value, ((0, 0), (self.left_padding, 0), (0, 0)))
        return self._convolve(value)


class SemanticRMSNorm(nn.Module):
    """Affine RMSNorm matching the semantic encoder's float32 epsilon."""

    def __init__(self, dimension: int):
        super().__init__()
        self.weight = mx.ones((dimension,))
        self.eps = 1.1920928955078125e-7

    def __call__(self, value: mx.array) -> mx.array:
        return mx.fast.rms_norm(value, self.weight, self.eps)


SemanticLayerCache = BoundedKVCache


class SemanticAttention(nn.Module):
    """Plain multi-head causal attention used by the semantic encoder."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("semantic hidden_size must divide evenly by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def fuse_qkv_for_inference(self) -> None:
        """Replace three projections with one weight-exact MLX matmul."""

        if getattr(self, "qkv_proj", None) is not None:
            return
        fused = _FusedLinear((self.q_proj, self.k_proj, self.v_proj))
        mx.eval(fused.parameters())
        self.qkv_proj = fused
        del self.q_proj
        del self.k_proj
        del self.v_proj

    @staticmethod
    def _mask(offset: int, length: int, *, dtype: mx.Dtype) -> mx.array | None:
        if length == 1:
            return None
        query = mx.arange(offset, offset + length, dtype=mx.int32)[:, None]
        key = mx.arange(offset + length, dtype=mx.int32)[None, :]
        allowed = query >= key
        return mx.where(allowed, 0.0, float("-inf")).astype(dtype)[None, None]

    def __call__(
        self,
        value: mx.array,
        *,
        cache: SemanticLayerCache | None = None,
        cache_capacity: int | None = None,
    ) -> tuple[mx.array, SemanticLayerCache]:
        batch, length, _ = value.shape
        qkv_proj = getattr(self, "qkv_proj", None)
        if qkv_proj is None:
            query = self.q_proj(value)
            keys = self.k_proj(value)
            values = self.v_proj(value)
        else:
            query, keys, values = mx.split(qkv_proj(value), 3, axis=-1)
        query = query.reshape(batch, length, self.num_heads, self.head_dim)
        keys = keys.reshape(batch, length, self.num_heads, self.head_dim)
        values = values.reshape(batch, length, self.num_heads, self.head_dim)
        offset = 0
        if cache is not None:
            cached_keys, _ = cache.fetch()
            if int(cached_keys.shape[0]) != batch:
                raise ValueError("semantic cache batch size differs from input")
            offset = cache.offset
            cache.append(keys, values)
            next_cache = cache
        else:
            capacity = length if cache_capacity is None else cache_capacity
            next_cache = SemanticLayerCache.from_values(
                keys,
                values,
                capacity=capacity,
                max_capacity=capacity,
            )
        keys, values = next_cache.fetch()
        mask = self._mask(offset, length, dtype=query.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attended), next_cache


class SemanticMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(value)))


class SemanticEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attn = SemanticAttention(hidden_size, num_heads)
        self.attn_norm = SemanticRMSNorm(hidden_size)
        self.ffn = SemanticMLP(hidden_size, intermediate_size)
        self.ffn_norm = SemanticRMSNorm(hidden_size)

    def __call__(
        self,
        value: mx.array,
        *,
        cache: SemanticLayerCache | None = None,
        cache_capacity: int | None = None,
    ) -> tuple[mx.array, SemanticLayerCache]:
        attended, next_cache = self.attn(
            self.attn_norm(value),
            cache=cache,
            cache_capacity=cache_capacity,
        )
        value = value + attended
        return value + self.ffn(self.ffn_norm(value)), next_cache


__all__ = [
    "CausalConv1d",
    "SemanticEncoderLayer",
    "SemanticLayerCache",
]
