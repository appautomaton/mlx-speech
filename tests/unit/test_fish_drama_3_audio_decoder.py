"""Drama 3 audio decoder and sampling against numpy references of the upstream rules.

Float checks run on the CPU, where float32 matmuls are exact enough to compare.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models._cache import BoundedKVCache
from mlx_speech.models.fish_drama_3 import FishDrama3Config
from mlx_speech.models.fish_drama_3.audio_decoder import (
    FishDrama3AudioDecoder,
    TowerLayer,
)
from mlx_speech.models.fish_drama_3.sampling import (
    resample_repeats,
    top_k_top_p_candidates,
)
from tests.unit.test_fish_drama_3_config import tiny_payload
from tests.unit.test_fish_drama_3_layers import (
    as_np,
    randomize,
    rms,
    run_in_chunks,
    sigmoid,
    silu,
    softmax,
)

CONFIG = FishDrama3Config.from_dict(tiny_payload())
DECODER = CONFIG.audio_decoder


@pytest.fixture(autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def upstream_kept(logits, temperature, top_k, top_p):
    """Ids and renormalized probabilities upstream may draw, best first."""
    scaled = logits / max(temperature, 1e-5)
    order = np.argsort(-scaled, kind="stable")
    probabilities = softmax(scaled[order])
    dropped = (np.cumsum(probabilities) > top_p) | (np.arange(len(order)) >= top_k)
    dropped[0] = False
    kept = probabilities[~dropped]
    return order[~dropped], kept / kept.sum()


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"),
    [
        (0.7, 30, 0.7),
        (1.0, 30, 0.95),
        (0.7, 5, 0.99),
        (1.0, 4097, 0.5),
        (1e-9, 30, 0.7),
    ],
)
def test_top_k_top_p_keeps_exactly_what_upstream_keeps(temperature, top_k, top_p):
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, 4097)

    ids, values = top_k_top_p_candidates(
        mx.array(logits[None], dtype=mx.float32), temperature, top_k, top_p
    )

    ids, values = np.array(ids[0]), as_np(values[0])
    kept = np.isfinite(values)
    expected_ids, expected_probabilities = upstream_kept(
        logits, temperature, top_k, top_p
    )
    np.testing.assert_array_equal(ids[kept], expected_ids)
    np.testing.assert_allclose(softmax(values[kept]), expected_probabilities, rtol=1e-4)


def test_repeated_semantic_draws_are_redrawn_but_a_stop_is_not():
    eos, semantic = 150, 210
    history = mx.array([eos] + [semantic] * 15 + [205])
    first = mx.array([semantic, eos, 230])
    retry = mx.array([999, 998, 997])
    is_semantic = (first >= 200) & (first <= 263)

    chosen = resample_repeats(first, retry, history, is_semantic)

    assert chosen.tolist() == [999, eos, 230]


def zero_centered(x, norm):
    return rms(x, norm.eps) * (1 + as_np(norm.weight))


def interleaved_rope(x, positions, width, base):
    angles = positions[:, None] * base ** (-np.arange(0, width, 2) / width)
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    even, odd = x[..., :width:2], x[..., 1:width:2]
    rotated = np.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    return np.concatenate([rotated.reshape(*x.shape[:-1], width), x[..., width:]], -1)


def tower_layer_oracle(layer: TowerLayer, hidden, codes, semantic, shared_input):
    embedding = as_np(layer.semantic_embedding.weight)
    embedded = np.where(semantic[:, None], embedding[codes], 0.0) + shared_input
    x = (
        np.concatenate(
            [zero_centered(embedded, layer.enorm), zero_centered(hidden, layer.hnorm)],
            -1,
        )
        @ as_np(layer.eh_proj.weight).T
    )

    block = layer.transformer_layer
    attention = block.attention
    heads, kv_heads, dim = attention.n_head, attention.n_local_heads, attention.head_dim
    length = len(x)
    q_width, kv_width = heads * dim, kv_heads * dim
    qkv = zero_centered(x, block.attention_norm) @ as_np(attention.wqkv.weight).T
    query, gate = qkv[:, :q_width], qkv[:, q_width : 2 * q_width]
    key, value = (
        qkv[:, 2 * q_width : 2 * q_width + kv_width],
        qkv[:, 2 * q_width + kv_width :],
    )
    positions = np.arange(length, dtype=np.float64)
    query = interleaved_rope(
        zero_centered(query.reshape(length, heads, dim), attention.q_norm),
        positions,
        attention.rotary_dim,
        attention.rope_base,
    )
    key = interleaved_rope(
        zero_centered(key.reshape(length, kv_heads, dim), attention.k_norm),
        positions,
        attention.rotary_dim,
        attention.rope_base,
    )
    value = value.reshape(length, kv_heads, dim)
    attended = np.zeros((length, heads, dim))
    causal = np.tril(np.ones((length, length), dtype=bool))
    for head in range(heads):
        kv = head // (heads // kv_heads)
        scores = np.where(causal, query[:, head] @ key[:, kv].T * dim**-0.5, -np.inf)
        attended[:, head] = softmax(scores) @ value[:, kv]
    x = (
        x
        + (attended.reshape(length, -1) * sigmoid(gate)) @ as_np(attention.wo.weight).T
    )
    mlp = block.feed_forward
    h = zero_centered(x, block.ffn_norm)
    x = (
        x
        + (silu(h @ as_np(mlp.w1.weight).T) * (h @ as_np(mlp.w3.weight).T))
        @ as_np(mlp.w2.weight).T
    )
    return zero_centered(x, layer.final_layernorm)


class _RowsAsBatch:
    """Adapts a TowerLayer to run_in_chunks, which slices [1, rows, dim] inputs."""

    def __init__(self, layer, codes, semantic, shared_input):
        self.layer, self.codes, self.semantic, self.shared_input = (
            layer,
            codes,
            semantic,
            shared_input,
        )
        self.start = 0

    def __call__(self, x, cache):
        rows = slice(self.start, self.start + x.shape[1])
        self.start += x.shape[1]
        return self.layer(
            x[0], self.codes[rows], self.semantic[rows], cache, self.shared_input[rows]
        )[None]


def test_tower_layer_in_cached_chunks_matches_oracle():
    rng = np.random.default_rng(1)
    layer = TowerLayer(DECODER, shared=True)
    randomize(layer, rng)
    length = 7
    hidden = rng.normal(0, 1, (length, DECODER.dim))
    codes = rng.integers(0, DECODER.vocab_size, length)
    semantic = np.array([True, True, False, True, True, True, False])
    shared_input = rng.normal(0, 0.1, (length, DECODER.dim))
    cache = BoundedKVCache.allocate(
        batch_size=1,
        capacity=16,
        num_heads=DECODER.n_local_heads,
        head_dim=DECODER.head_dim,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )
    adapter = _RowsAsBatch(
        layer,
        mx.array(codes),
        mx.array(semantic),
        mx.array(shared_input, dtype=mx.float32),
    )

    got = run_in_chunks(
        adapter, mx.array(hidden[None], dtype=mx.float32), cache, [4, 1, 2]
    )

    expected = tower_layer_oracle(layer, hidden, codes, semantic, shared_input)
    np.testing.assert_allclose(as_np(got[0]), expected, rtol=1e-4, atol=1e-5)


def test_frame_embeddings_mix_codes_as_upstream():
    rng = np.random.default_rng(2)
    decoder = FishDrama3AudioDecoder(CONFIG)
    randomize(decoder, rng)
    vocab = DECODER.vocab_size
    frames = np.array([[17, 0, 0, 0], [205, 5, 9, 63], [263, 63, 0, 12]])
    text = rng.normal(0, 1, (3, DECODER.dim))
    table = as_np(decoder.codebook_embeddings.weight)

    backbone_input = decoder.embed_backbone(
        mx.array(frames), mx.array(text, dtype=mx.float32)
    )
    unseen = decoder.embedding_sum(mx.array(frames), 2, DECODER.num_codebooks)

    expected = text.copy()
    expected[1:] = (text[1:] + table[frames[1:, 2] + vocab]) / np.sqrt(2)
    np.testing.assert_allclose(as_np(backbone_input), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        as_np(unseen)[1:], table[frames[1:, 3] + 2 * vocab], rtol=1e-6
    )
