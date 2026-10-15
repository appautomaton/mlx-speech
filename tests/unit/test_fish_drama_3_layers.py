"""Drama 3 backbone layers against float64 numpy oracles written from the upstream math.

Oracle checks run on the CPU: the M5 GPU computes float32 matmuls at about 1e-3
relative precision, which would hide real errors. One check per layer type runs the
production path (Metal kernels, cached chunks) against the CPU at bf16-level tolerance.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models._cache import BoundedKVCache
from mlx_speech.models.fish_drama_3 import ExpertQuantization, FishDrama3Config
from mlx_speech.models.fish_drama_3.layers import (
    DecoderLayer,
    DeltaNetState,
    GatedAttention,
    GatedDeltaNet,
    SparseMoeBlock,
)
from tests.unit.test_fish_drama_3_config import tiny_payload

CONFIG = FishDrama3Config.from_dict(tiny_payload()).text
QUANT = ExpertQuantization(bits=8, group_size=64)


@pytest.fixture(autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def as_np(array: mx.array) -> np.ndarray:
    return np.array(array.astype(mx.float32), dtype=np.float64)


def randomize(module, rng: np.random.Generator, scale: float = 0.1) -> None:
    """Give every float parameter random values, so zero-initialized norms count."""
    for name, value in _float_leaves(module.parameters()):
        _assign(
            module, name, mx.array(rng.normal(0, scale, value.shape), dtype=value.dtype)
        )


def _float_leaves(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _float_leaves(value, f"{prefix}{key}.")
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            yield from _float_leaves(value, f"{prefix}{index}.")
    elif mx.issubdtype(tree.dtype, mx.floating):
        yield prefix[:-1], tree


def _assign(module, dotted: str, value: mx.array) -> None:
    *path, leaf = dotted.split(".")
    target = module
    for part in path:
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    setattr(target, leaf, value)


def rms(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def neox_rope(x: np.ndarray, positions: np.ndarray, width: int, theta: float):
    """x [T, heads, dim]: rotate the first width channels, half-split."""
    angles = positions[:, None] * theta ** (-np.arange(0, width, 2) / width)
    cos = np.cos(np.concatenate([angles, angles], -1))[:, None, :]
    sin = np.sin(np.concatenate([angles, angles], -1))[:, None, :]
    rotary = x[..., :width]
    first, second = rotary[..., : width // 2], rotary[..., width // 2 :]
    rotated = rotary * cos + np.concatenate([-second, first], -1) * sin
    return np.concatenate([rotated, x[..., width:]], -1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def attention_oracle(module: GatedAttention, x: np.ndarray) -> np.ndarray:
    """x [T, hidden] from position 0."""
    heads, kv_heads, dim = module.num_heads, module.num_kv_heads, module.head_dim
    eps = module.q_norm.eps
    length = len(x)
    q_full = (x @ as_np(module.q_proj.weight).T).reshape(length, heads, 2 * dim)
    query, gate = q_full[..., :dim], q_full[..., dim:].reshape(length, heads * dim)
    query = rms(query, eps) * (1 + as_np(module.q_norm.weight))
    key = (x @ as_np(module.k_proj.weight).T).reshape(length, kv_heads, dim)
    key = rms(key, eps) * (1 + as_np(module.k_norm.weight))
    value = (x @ as_np(module.v_proj.weight).T).reshape(length, kv_heads, dim)
    positions = np.arange(length, dtype=np.float64)
    query = neox_rope(query, positions, module.rotary_dim, module.rope_theta)
    key = neox_rope(key, positions, module.rotary_dim, module.rope_theta)
    out = np.zeros((length, heads, dim))
    causal = np.tril(np.ones((length, length), dtype=bool))
    for head in range(heads):
        kv = head // (heads // kv_heads)
        scores = np.where(causal, query[:, head] @ key[:, kv].T * dim**-0.5, -np.inf)
        out[:, head] = softmax(scores) @ value[:, kv]
    return (out.reshape(length, -1) * sigmoid(gate)) @ as_np(module.o_proj.weight).T


def deltanet_oracle(module: GatedDeltaNet, x: np.ndarray) -> np.ndarray:
    """x [T, hidden] from an empty state, token by token as upstream's recurrence."""
    length = len(x)
    kernel = module.kernel_size
    mixed = x @ as_np(module.in_proj_qkv.weight).T
    padded = np.concatenate([np.zeros((kernel - 1, module.conv_dim)), mixed])
    weight = as_np(module.conv1d.weight)[:, :, 0]
    conv = sum(padded[tap : tap + length] * weight[:, tap] for tap in range(kernel))
    mixed = silu(conv)
    kd, hk, hv = module.key_head_dim, module.num_key_heads, module.num_value_heads
    vd = module.value_head_dim
    query = mixed[:, : module.key_dim].reshape(length, hk, kd)
    key = mixed[:, module.key_dim : 2 * module.key_dim].reshape(length, hk, kd)
    value = mixed[:, 2 * module.key_dim :].reshape(length, hv, vd)
    z = (x @ as_np(module.in_proj_z.weight).T).reshape(length, hv, vd)
    beta = sigmoid(x @ as_np(module.in_proj_b.weight).T)
    softplus = np.log1p(
        np.exp(x @ as_np(module.in_proj_a.weight).T + as_np(module.dt_bias))
    )
    decay = -np.exp(as_np(module.A_log)) * softplus

    def l2(a):
        return a / np.sqrt((a**2).sum(-1, keepdims=True) + 1e-6)

    query = np.repeat(l2(query) * kd**-0.5, hv // hk, axis=1)
    key = np.repeat(l2(key), hv // hk, axis=1)
    state = np.zeros((hv, kd, vd))
    core = np.zeros((length, hv, vd))
    for t in range(length):
        state = state * np.exp(decay[t])[:, None, None]
        remembered = (state * key[t][:, :, None]).sum(1)
        delta = (value[t] - remembered) * beta[t][:, None]
        state = state + key[t][:, :, None] * delta[:, None, :]
        core[t] = (state * query[t][:, :, None]).sum(1)
    normed = as_np(module.norm.weight) * rms(core, module.norm.eps) * silu(z)
    return normed.reshape(length, -1) @ as_np(module.out_proj.weight).T


def moe_oracle(module: SparseMoeBlock, x: np.ndarray) -> np.ndarray:
    def dense(proj):
        return as_np(
            mx.dequantize(
                proj.weight,
                proj.scales,
                proj.biases,
                group_size=proj.group_size,
                bits=proj.bits,
            )
        )

    experts = module.experts
    gate_w, up_w, down_w = (
        dense(p) for p in (experts.gate_proj, experts.up_proj, experts.down_proj)
    )
    probabilities = softmax(x @ as_np(module.gate.weight).T)
    out = np.zeros_like(x)
    for row in range(len(x)):
        ids = np.argsort(-probabilities[row])[: module.top_k]
        weights = probabilities[row, ids] / probabilities[row, ids].sum()
        for expert, weight in zip(ids, weights):
            hidden = silu(gate_w[expert] @ x[row]) * (up_w[expert] @ x[row])
            out[row] += weight * (down_w[expert] @ hidden)
    shared = module.shared_expert
    shared_out = (
        silu(x @ as_np(shared.gate_proj.weight).T)
        * (x @ as_np(shared.up_proj.weight).T)
    ) @ as_np(shared.down_proj.weight).T
    return out + sigmoid(x @ as_np(module.shared_expert_gate.weight).T) * shared_out


def random_moe(rng: np.random.Generator) -> SparseMoeBlock:
    module = SparseMoeBlock(CONFIG, QUANT)
    randomize(module, rng)
    for proj in (
        module.experts.gate_proj,
        module.experts.up_proj,
        module.experts.down_proj,
    ):
        experts, rows, packed = proj.weight.shape
        dense = mx.array(
            rng.normal(0, 0.05, (experts, rows, packed * 32 // QUANT.bits)),
            dtype=mx.float32,
        )
        proj.weight, proj.scales, proj.biases = mx.quantize(
            dense, group_size=QUANT.group_size, bits=QUANT.bits
        )
    return module


def kv_cache(capacity: int = 128) -> BoundedKVCache:
    return BoundedKVCache.allocate(
        batch_size=1,
        capacity=capacity,
        num_heads=CONFIG.num_key_value_heads,
        head_dim=CONFIG.head_dim,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )


def delta_state() -> DeltaNetState:
    return DeltaNetState.allocate(CONFIG, batch_size=1, dtype=mx.float32)


def run_in_chunks(module, x: mx.array, cache, sizes: list[int]) -> mx.array:
    outputs, start = [], 0
    for size in sizes:
        outputs.append(module(x[:, start : start + size], cache))
        start += size
    return mx.concatenate(outputs, axis=1)


def test_gated_attention_matches_oracle():
    rng = np.random.default_rng(2)
    module = GatedAttention(CONFIG)
    randomize(module, rng)
    x = rng.normal(0, 1, (7, CONFIG.hidden_size))

    got = module(mx.array(x[None], dtype=mx.float32), kv_cache())

    np.testing.assert_allclose(
        as_np(got[0]), attention_oracle(module, x), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("length", [9, 150])  # sequential and chunked kernels
def test_gated_deltanet_matches_recurrent_oracle(length):
    rng = np.random.default_rng(4)
    module = GatedDeltaNet(CONFIG)
    randomize(module, rng, scale=0.2)
    x = rng.normal(0, 1, (length, CONFIG.hidden_size))

    got = module(mx.array(x[None], dtype=mx.float32), delta_state())

    np.testing.assert_allclose(
        as_np(got[0]), deltanet_oracle(module, x), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("rows", [3, 40])  # 6 pairs stay in token order; 80 get sorted
def test_moe_matches_dense_oracle(rows):
    rng = np.random.default_rng(6)
    module = random_moe(rng)
    x = rng.normal(0, 1, (1, rows, CONFIG.hidden_size))

    got = module(mx.array(x, dtype=mx.float32))

    np.testing.assert_allclose(
        as_np(got[0]), moe_oracle(module, x[0]), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("layer_index", [0, 1])  # Gated DeltaNet, then gated attention
def test_decoder_layer_on_gpu_in_cached_chunks_matches_one_cpu_pass(layer_index):
    rng = np.random.default_rng(7)
    layer = DecoderLayer(CONFIG, layer_index, QUANT)
    randomize(layer, rng)
    layer.mlp = random_moe(rng)
    x = mx.array(rng.normal(0, 1, (1, 80, CONFIG.hidden_size)), dtype=mx.float32)

    def cache():
        return delta_state() if layer.is_linear else kv_cache()

    reference = layer(x, cache())
    mx.eval(reference)
    with mx.stream(mx.gpu):
        chunked = run_in_chunks(layer, x, cache(), [70, 1, 5, 1, 3])

    error = np.linalg.norm(as_np(chunked) - as_np(reference)) / np.linalg.norm(
        as_np(reference)
    )
    assert error < 5e-3
