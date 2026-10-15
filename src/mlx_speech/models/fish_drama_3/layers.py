"""Drama 3 backbone layers: the Qwen3.5-MoE hybrid of gated softmax attention and
Gated DeltaNet, each followed by a routed mixture of experts."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .._cache import BoundedKVCache
from .config import ExpertQuantization, FishDrama3TextConfig

# From this many (token, expert) pairs on, the experts run on expert-sorted rows,
# which lets gather_qmm reuse each expert's weights across its rows (prefill).
SORTED_EXPERT_MIN_PAIRS = 64
L2NORM_EPS = 1e-6


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        scale = 1.0 + self.weight.astype(mx.float32)
        return mx.fast.rms_norm(x.astype(mx.float32), scale, self.eps).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    """Gated DeltaNet output norm: weight * rmsnorm(x) * silu(gate).

    Unlike every other norm in the model, the weight is used as is, not as 1 + weight.
    """

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps)
        out = self.weight * normed.astype(x.dtype)
        return (out * nn.silu(gate.astype(mx.float32))).astype(x.dtype)


class GatedAttention(nn.Module):
    """Softmax attention: GQA, q/k norm, partial NeoX RoPE, sigmoid output gate."""

    def __init__(self, config: FishDrama3TextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        # Output rows hold each head's [query, gate] pair.
        self.q_proj = nn.Linear(hidden, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.rotary_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    def __call__(self, x: mx.array, cache: BoundedKVCache) -> mx.array:
        batch, length, _ = x.shape
        offset = cache.offset
        query, gate = mx.split(
            self.q_proj(x).reshape(batch, length, self.num_heads, 2 * self.head_dim),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, self.num_heads * self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        )
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        query = self.rope(query.transpose(0, 2, 1, 3), offset)
        key = self.rope(key.transpose(0, 2, 1, 3), offset)

        # The cache keeps [batch, positions, heads, dim]; attention reads it through
        # a transposed view, which costs nothing extra.
        cache.append(key.transpose(0, 2, 1, 3), value)
        keys, values = cache.fetch()
        attended = mx.fast.scaled_dot_product_attention(
            query,
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
            mask="causal" if length > 1 else None,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, length, self.num_heads * self.head_dim
        )
        return self.o_proj(attended * mx.sigmoid(gate))


@dataclass
class DeltaNetState:
    """Recurrent state of one Gated DeltaNet layer."""

    conv: mx.array  # [batch, kernel - 1, channels]: the newest depthwise-conv inputs
    recurrent: mx.array  # [batch, value_heads, value_dim, key_dim], fp32

    @classmethod
    def allocate(
        cls, config: FishDrama3TextConfig, *, batch_size: int, dtype: mx.Dtype
    ) -> DeltaNetState:
        channels = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        return cls(
            conv=mx.zeros(
                (batch_size, config.linear_conv_kernel_dim - 1, channels), dtype=dtype
            ),
            recurrent=mx.zeros(
                (
                    batch_size,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                    config.linear_key_head_dim,
                ),
                dtype=mx.float32,
            ),
        )


def l2norm(x: mx.array) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + L2NORM_EPS)


class GatedDeltaNet(nn.Module):
    """Linear attention by the gated delta rule, after a causal depthwise conv."""

    def __init__(self, config: FishDrama3TextConfig):
        super().__init__()
        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel_size = config.linear_conv_kernel_dim
        # Output rows: query, key, value.
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = mx.ones((self.num_value_heads,))
        self.A_log = mx.zeros((self.num_value_heads,))
        self.norm = GatedRMSNorm(self.value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def causal_conv(self, mixed: mx.array, state: DeltaNetState) -> mx.array:
        """Depthwise conv over the cached history plus mixed, accumulated in fp32."""
        length = mixed.shape[1]
        padded = mx.concatenate([state.conv.astype(mixed.dtype), mixed], axis=1)
        state.conv = padded[:, -(self.kernel_size - 1) :, :]
        weight = self.conv1d.weight[:, :, 0].astype(mx.float32)  # [channels, kernel]
        out = padded[:, :length, :].astype(mx.float32) * weight[:, 0]
        for tap in range(1, self.kernel_size):
            out = (
                out
                + padded[:, tap : tap + length, :].astype(mx.float32) * weight[:, tap]
            )
        return out.astype(mixed.dtype)

    def __call__(self, x: mx.array, state: DeltaNetState) -> mx.array:
        batch, length, _ = x.shape
        mixed = nn.silu(self.causal_conv(self.in_proj_qkv(x), state))
        query, key, value = mx.split(mixed, [self.key_dim, 2 * self.key_dim], axis=-1)
        query = query.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, length, self.num_value_heads, self.value_head_dim)
        z = self.in_proj_z(x).reshape(
            batch, length, self.num_value_heads, self.value_head_dim
        )
        beta = mx.sigmoid(self.in_proj_b(x).astype(mx.float32))
        decay = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(
            self.in_proj_a(x).astype(mx.float32) + self.dt_bias.astype(mx.float32)
        )
        # The kernel takes normalized, scaled queries and keys with key heads shared
        # by consecutive value heads, and the decay in linear space.
        query = l2norm(query.astype(mx.float32)) * self.key_head_dim**-0.5
        key = l2norm(key.astype(mx.float32))
        core, state.recurrent = mx.fast.gated_delta_update(
            query,
            key,
            value.astype(mx.float32),
            mx.exp(decay),
            beta,
            state.recurrent,
        )
        core = self.norm(
            core.astype(x.dtype).reshape(-1, self.value_head_dim),
            z.reshape(-1, self.value_head_dim),
        )
        return self.out_proj(core.reshape(batch, length, self.value_dim))


class QuantizedSwitchLinear(nn.Module):
    """Every expert's linear layer, stacked and quantized: weight [experts, out, in]."""

    def __init__(
        self,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        quantization: ExpertQuantization,
    ):
        super().__init__()
        if quantization.mode != "affine":
            raise NotImplementedError("only affine expert quantization is supported")
        self.group_size = quantization.group_size
        self.bits = quantization.bits
        groups = input_dims // quantization.group_size
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * quantization.bits // 32),
            dtype=mx.uint32,
        )
        self.scales = mx.zeros((num_experts, output_dims, groups))
        self.biases = mx.zeros((num_experts, output_dims, groups))

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            sorted_indices=sorted_indices,
        )


class RoutedExperts(nn.Module):
    """SwiGLU experts; each row runs only through the experts routed to it."""

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        intermediate: int,
        quantization: ExpertQuantization,
    ):
        super().__init__()
        self.gate_proj = QuantizedSwitchLinear(
            num_experts, hidden, intermediate, quantization
        )
        self.up_proj = QuantizedSwitchLinear(
            num_experts, hidden, intermediate, quantization
        )
        self.down_proj = QuantizedSwitchLinear(
            num_experts, intermediate, hidden, quantization
        )

    def __call__(self, x: mx.array, ids: mx.array, weights: mx.array) -> mx.array:
        """x [rows, hidden], ids / weights [rows, k] -> weighted expert sum [rows, hidden]."""
        rows, top_k = ids.shape
        expanded = x[:, None, None, :]  # [rows, 1, 1, hidden]
        sort = ids.size >= SORTED_EXPERT_MIN_PAIRS
        if sort:
            flat_ids = ids.flatten()
            order = mx.argsort(flat_ids)
            expanded = expanded.flatten(0, -3)[order // top_k]  # [pairs, 1, hidden]
            ids = flat_ids[order]
        gate = self.gate_proj(expanded, ids, sorted_indices=sort)
        up = self.up_proj(expanded, ids, sorted_indices=sort)
        out = self.down_proj(nn.silu(gate) * up, ids, sorted_indices=sort)
        if sort:
            out = out[mx.argsort(order)].reshape(rows, top_k, 1, -1)
        out = out.squeeze(-2).astype(mx.float32) * weights[..., None].astype(mx.float32)
        return out.sum(axis=1).astype(x.dtype)


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, config: FishDrama3TextConfig, quantization: ExpertQuantization):
        super().__init__()
        hidden = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(hidden, config.num_experts, bias=False)
        self.experts = RoutedExperts(
            config.num_experts, hidden, config.moe_intermediate_size, quantization
        )
        self.shared_expert = DenseMLP(hidden, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        probabilities = mx.softmax(self.gate(flat).astype(mx.float32), axis=-1)
        ids = mx.argpartition(-probabilities, kth=self.top_k - 1, axis=-1)[
            :, : self.top_k
        ]
        weights = mx.take_along_axis(probabilities, ids, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        # Upstream rounds the routing weights to the activation dtype first.
        routed = self.experts(flat, ids, weights.astype(flat.dtype))
        shared = mx.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return (routed + shared).reshape(x.shape)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: FishDrama3TextConfig,
        layer_index: int,
        quantization: ExpertQuantization,
    ):
        super().__init__()
        self.is_linear = config.layer_types[layer_index] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.mlp = SparseMoeBlock(config, quantization)
        self.input_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x: mx.array, cache: BoundedKVCache | DeltaNetState) -> mx.array:
        mixer = self.linear_attn if self.is_linear else self.self_attn
        x = x + mixer(self.input_layernorm(x), cache)
        return x + self.mlp(self.post_attention_layernorm(x))
