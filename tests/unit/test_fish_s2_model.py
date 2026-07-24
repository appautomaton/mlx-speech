import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.fish_s2_pro.config import (
    FishAudioDecoderConfig,
    FishS2ProConfig,
    FishTextConfig,
)
from mlx_speech.models.fish_s2_pro.cache import KVCache
from mlx_speech.models.fish_s2_pro.model import DualARTransformer


def _tiny_config():
    return FishS2ProConfig(
        text_config=FishTextConfig(
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_local_heads=2,
            head_dim=8,
            dim=16,
            intermediate_size=32,
            max_seq_len=32,
        ),
        audio_decoder_config=FishAudioDecoderConfig(
            vocab_size=8,
            n_layer=2,
            n_head=2,
            n_local_heads=2,
            head_dim=8,
            dim=16,
            intermediate_size=32,
            max_seq_len=4,
            text_dim=16,
            num_codebooks=3,
        ),
        semantic_start_token_id=20,
        semantic_end_token_id=27,
    )


def test_dual_ar_forward_returns_logits_and_hidden_states():
    model = DualARTransformer(_tiny_config())
    x = mx.zeros((1, 4, 3), dtype=mx.int32)
    result = model(x)
    assert result.logits.shape == (1, 3, 32)
    assert result.hidden_states.shape == (1, 3, 16)


def test_dual_ar_forward_rejects_wrong_row_count():
    model = DualARTransformer(_tiny_config())
    x = mx.zeros((1, 3, 3), dtype=mx.int32)

    with pytest.raises(ValueError, match=r"Expected input shape \(batch, 4, seq\)"):
        model(x)


def test_codebook_rows_use_distinct_embedding_offsets():
    config = _tiny_config()
    config.text_config.n_layer = 0
    model = DualARTransformer(config)
    model.embeddings.weight = mx.zeros_like(model.embeddings.weight)
    model.codebook_embeddings.weight = mx.zeros_like(model.codebook_embeddings.weight)
    model.codebook_embeddings.weight[1] = mx.ones(
        (config.text_config.dim,), dtype=mx.float32
    )
    model.codebook_embeddings.weight[config.audio_decoder_config.vocab_size + 1] = (
        2 * mx.ones((config.text_config.dim,), dtype=mx.float32)
    )

    semantic = config.semantic_start_token_id
    row0 = mx.array([[[semantic], [1], [0], [0]]], dtype=mx.int32)
    row1 = mx.array([[[semantic], [0], [1], [0]]], dtype=mx.int32)

    embed0 = model._embed(row0)
    embed1 = model._embed(row1)

    assert mx.any(embed0 != embed1).item()


def test_fast_forward_returns_codebook_logits_with_previous_codebooks():
    model = DualARTransformer(_tiny_config())
    hidden = mx.zeros((1, 1, 16), dtype=mx.float32)
    prev = mx.array([[1, 2]], dtype=mx.int32)
    logits = model.fast_forward(hidden, prev)
    assert logits.shape == (1, 8)


def test_model_parameters_use_upstream_fused_projection_names():
    model = DualARTransformer(_tiny_config())
    params = tree_flatten(model.parameters(), destination={})
    keys = set(params)

    expected_keys = {
        "layers.0.attention.wqkv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.attention.k_norm.weight",
        "fast_layers.0.attention.wqkv.weight",
        "fast_layers.0.attention.wo.weight",
        "fast_layers.0.feed_forward.w1.weight",
        "fast_layers.0.feed_forward.w2.weight",
        "fast_layers.0.feed_forward.w3.weight",
    }
    assert expected_keys <= keys
    assert "fast_layers.0.attention.q_norm.weight" not in keys
    assert "fast_layers.0.attention.k_norm.weight" not in keys

    forbidden_fragments = (
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".gate_proj.",
        ".up_proj.",
        ".down_proj.",
    )
    assert not any(fragment in key for key in keys for fragment in forbidden_fragments)


def test_kv_cache_preserves_dtype_and_requires_initialization():
    cache = KVCache(num_layers=2, dim=16, max_length=8)

    with pytest.raises(RuntimeError, match="KV cache is uninitialized"):
        cache.get(0)

    key = mx.zeros((1, 2, 3, 8), dtype=mx.bfloat16)
    value = mx.ones((1, 2, 3, 8), dtype=mx.bfloat16)
    cache.update(0, key, value)

    keys, values = cache.get(0)
    assert keys.dtype == mx.bfloat16
    assert values.dtype == mx.bfloat16
    assert keys.shape == (1, 2, 3, 8)
    assert values.shape == (1, 2, 3, 8)


def test_kv_cache_layers_use_independent_buffers():
    """Layer buffers must not alias. A shared tensor made decode allocate GBs per token."""
    cache = KVCache(num_layers=3, dim=16, max_length=64)

    for layer_idx in range(3):
        step = mx.full((1, 2, 1, 8), float(layer_idx + 1), dtype=mx.float32)
        cache.update(layer_idx, step, step)

    buffers = {id(cache._keys[i]) for i in range(3)}
    assert len(buffers) == 3

    for layer_idx in range(3):
        keys, _ = cache.get(layer_idx)
        assert float(keys[0, 0, 0, 0]) == float(layer_idx + 1)


def test_kv_cache_preserves_history_across_chunk_growth():
    """Token 257 forces a realloc. Everything written before it must survive."""
    cache = KVCache(num_layers=1, dim=16, max_length=1024)

    written = []
    for step in range(300):
        token = mx.full((1, 2, 1, 8), float(step), dtype=mx.float32)
        cache.update(0, token, token)
        written.append(float(step))

    keys, values = cache.get(0)
    assert keys.shape == (1, 2, 300, 8)
    assert cache._keys[0].shape[2] > 256, "expected the buffer to have grown past one chunk"

    recovered = [float(keys[0, 0, i, 0]) for i in range(300)]
    assert recovered == written
    assert [float(values[0, 0, i, 0]) for i in range(300)] == written


def test_kv_cache_reset_reuses_storage_and_trim_to_rewinds():
    cache = KVCache(num_layers=1, dim=16, max_length=64)

    token = mx.ones((1, 2, 8, 8), dtype=mx.float32)
    cache.update(0, token, token)
    buffer_id = id(cache._keys[0])

    cache.trim_to(3)
    assert cache.get(0)[0].shape[2] == 3

    cache.reset()
    assert cache.offset == 0
    cache.update(0, token, token)
    assert id(cache._keys[0]) == buffer_id, "reset should keep allocated storage"
    assert cache.get(0)[0].shape[2] == 8


def test_kv_cache_rejects_overflow_past_max_length():
    cache = KVCache(num_layers=1, dim=16, max_length=4)

    exact = mx.zeros((1, 2, 4, 8), dtype=mx.float32)
    cache.update(0, exact, exact)

    one_more = mx.zeros((1, 2, 1, 8), dtype=mx.float32)
    with pytest.raises(ValueError, match="KV cache overflow"):
        cache.update(0, one_more, one_more)
