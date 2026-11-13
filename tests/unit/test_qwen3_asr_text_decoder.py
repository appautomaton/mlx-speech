from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.config import Qwen3ASRTextConfig
from mlx_speech.models.qwen3_asr.text_decoder import (
    Qwen3ASRTextAttention,
    Qwen3ASRTextForCausalLM,
    Qwen3ASRTextKVCache,
    Qwen3ASRTextModel,
    Qwen3ASRTextRMSNorm,
    Qwen3ASRTextRotaryEmbedding,
    _make_additive_attention_mask,
    _repeat_kv,
    _sliding_window_attention,
)


def _tiny_config(**overrides) -> Qwen3ASRTextConfig:
    values = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_cache": True,
        "extra": {"tie_word_embeddings": True},
    }
    values.update(overrides)
    return Qwen3ASRTextConfig(**values)


def test_qwen3_asr_text_attention_shape_and_qk_norm():
    config = _tiny_config()
    attention = Qwen3ASRTextAttention(config, layer_idx=0)
    hidden = mx.zeros((2, 5, 16), dtype=mx.float32)

    output = attention(hidden)
    mx.eval(output)

    assert output.shape == (2, 5, 16)
    assert attention.kv_repeat == 2
    assert isinstance(attention.q_norm, Qwen3ASRTextRMSNorm)
    assert isinstance(attention.k_norm, Qwen3ASRTextRMSNorm)
    assert attention.q_norm.weight.shape == (4,)
    assert attention.k_norm.weight.shape == (4,)
    assert mx.all(mx.isfinite(output)).item()


def test_window_bounded_attention_matches_dense_reference_across_boundary():
    mx.random.seed(71)
    query = mx.random.normal((1, 4, 70, 4))
    key = mx.random.normal((1, 2, 70, 4))
    value = mx.random.normal((1, 2, 70, 4))
    scale = 0.5
    actual = _sliding_window_attention(
        query,
        key,
        value,
        scale=scale,
        sliding_window=64,
        query_block_size=32,
    )

    repeated_key = _repeat_kv(key, 2)
    repeated_value = _repeat_kv(value, 2)
    scores = mx.matmul(query, repeated_key.transpose(0, 1, 3, 2)) * scale
    mask = _make_additive_attention_mask(
        query_len=70,
        key_len=70,
        query_offset=0,
        dtype=mx.float32,
        use_causal_mask=True,
        sliding_window=64,
    )
    expected = mx.matmul(mx.softmax(scores + mask, axis=-1), repeated_value)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.asarray(actual)[:, :, [0, 63, 64, 69]],
        np.asarray(expected)[:, :, [0, 63, 64, 69]],
        atol=2e-5,
        rtol=2e-5,
    )


def test_qwen3_asr_text_kv_cache_prefill_and_decode_append():
    config = _tiny_config()
    model = Qwen3ASRTextModel(config)
    cache = Qwen3ASRTextKVCache.allocate(
        config,
        batch_size=1,
        max_length=8,
        dtype=mx.float32,
    )

    prefill = model.prefill(
        input_ids=mx.array([[1, 2, 3]], dtype=mx.int32),
        kv_cache=cache,
    )
    decode = model.decode_step(
        input_ids=mx.array([[4]], dtype=mx.int32),
        kv_cache=cache,
    )
    mx.eval(prefill.last_hidden_state, decode.last_hidden_state)

    assert cache.current_length == 4
    assert all(layer.current_length == 4 for layer in cache.layers)
    assert prefill.last_hidden_state.shape == (1, 3, 16)
    assert decode.last_hidden_state.shape == (1, 1, 16)


def test_qwen3_asr_text_kv_cache_rejects_overflow():
    config = _tiny_config(num_hidden_layers=1)
    cache = Qwen3ASRTextKVCache.allocate(
        config,
        batch_size=1,
        max_length=1,
        dtype=mx.float32,
    )
    key = mx.zeros((1, 2, 2, 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="overflow"):
        cache.layers[0].append(key, key)


def test_qwen3_asr_text_rope_uses_position_offset():
    rope = Qwen3ASRTextRotaryEmbedding(_tiny_config())

    cos0, sin0 = rope.cos_sin(seq_len=3, offset=0)
    cos1, sin1 = rope.cos_sin(seq_len=3, offset=1)
    mx.eval(cos0, sin0, cos1, sin1)

    assert cos0.shape == (3, 4)
    assert sin0.shape == (3, 4)
    assert not mx.allclose(cos0, cos1).item()
    assert not mx.allclose(sin0, sin1).item()


def test_qwen3_asr_text_lm_logits_shape_and_tied_embeddings():
    config = _tiny_config()
    model = Qwen3ASRTextForCausalLM(config)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    output = model(input_ids=input_ids)
    mx.eval(output.logits)

    assert output.logits.shape == (1, 4, 32)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qwen3_asr_text_lm_supports_untied_embeddings():
    config = _tiny_config(extra={"tie_word_embeddings": False})
    model = Qwen3ASRTextForCausalLM(config)

    assert model.lm_head.weight is not model.model.embed_tokens.weight


def test_qwen3_asr_text_parameter_names_match_checkpoint_surface():
    model = Qwen3ASRTextForCausalLM(_tiny_config())
    params = tree_flatten(model.parameters(), destination={})

    expected = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    }

    assert expected <= set(params)


def test_qwen3_asr_text_model_rejects_ambiguous_inputs():
    model = Qwen3ASRTextModel(_tiny_config())
    input_ids = mx.array([[1, 2]], dtype=mx.int32)
    embeds = mx.zeros((1, 2, 16), dtype=mx.float32)

    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=input_ids, inputs_embeds=embeds)

    with pytest.raises(ValueError, match="exactly one"):
        model()


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_left_padded_prefill_is_finite(fast, window):
    from mlx_speech.models.qwen3_asr.text_decoder import FAST_DECODE_COMPUTE

    mx.random.seed(17)
    extra = {} if window is None else {
        "use_sliding_window": True, "sliding_window": window, "max_window_layers": 0
    }
    config = _tiny_config(extra=extra)
    model = Qwen3ASRTextForCausalLM(config)
    cache = Qwen3ASRTextKVCache.allocate(config, batch_size=2, max_length=8)
    output = model.prefill(
        input_ids=mx.array([[0, 0, 1, 2], [1, 2, 3, 4]]),
        attention_mask=mx.array([[0, 0, 1, 1], [1, 1, 1, 1]]),
        kv_cache=cache,
        compute=FAST_DECODE_COMPUTE if fast else None,
    )
    assert bool(mx.all(mx.isfinite(output.logits)).item())
    for layer in cache.layers:
        assert bool(mx.all(mx.isfinite(layer.get()[0])).item())


def test_left_padding_mask_is_finite_and_unblocks_discarded_queries():
    mask = _make_additive_attention_mask(
        query_len=4, key_len=4, query_offset=0, dtype=mx.float32,
        attention_mask=mx.array([[0, 0, 1, 1]]), use_causal_mask=True,
    )
    assert bool(mx.all(mx.isfinite(mask)).item())
    np.testing.assert_array_equal(np.asarray(mask)[0, 0, :2], np.zeros((2, 4)))


@pytest.mark.parametrize("offset", [0, 7])
def test_explicit_rope_positions_match_scalar_offset_exactly(offset):
    mx.random.seed(19)
    rope = Qwen3ASRTextRotaryEmbedding(_tiny_config())
    positions = mx.broadcast_to(mx.arange(3)[None, :] + offset, (2, 3))
    scalar = rope.cos_sin(seq_len=3, offset=offset)
    batched = rope.cos_sin(seq_len=3, position_ids=positions)
    for expected, actual in zip(scalar, batched, strict=True):
        assert actual.shape == (2, 1, 3, 4)
        np.testing.assert_array_equal(np.asarray(actual), np.broadcast_to(expected, (2, 1, 3, 4)))
    query = mx.random.normal((2, 4, 3, 4))
    key = mx.random.normal((2, 2, 3, 4))
    for expected, actual in zip(
        rope.apply(query, key, offset=offset),
        rope.apply(query, key, position_ids=positions), strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_left_padded_decoder_matches_each_row_through_decode(fast, window):
    # A float32 oracle: Metal's optional TF32 GEMM changes rounding with B.
    with mx.stream(mx.cpu):
        from mlx_speech.models.qwen3_asr.text_decoder import FAST_DECODE_COMPUTE

        mx.random.seed(23)
        extra = {} if window is None else {
            "use_sliding_window": True, "sliding_window": window, "max_window_layers": 0
        }
        config = _tiny_config(extra=extra)
        model = Qwen3ASRTextForCausalLM(config)
        compute = FAST_DECODE_COMPUTE if fast else None
        sequences = [[1, 2], [3, 4, 5, 6, 7]]
        caches = [Qwen3ASRTextKVCache.allocate(config, batch_size=1, max_length=12) for _ in sequences]
        batch_cache = Qwen3ASRTextKVCache.allocate(config, batch_size=2, max_length=12)
        mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
        positions = mx.maximum(mx.arange(5)[None, :] - mx.array([3, 0])[:, None], 0)
        batch = model.prefill(
            input_ids=mx.array([[0, 0, 0, 1, 2], sequences[1]]), attention_mask=mask,
            position_ids=positions, kv_cache=batch_cache, compute=compute,
        )
        for row, (ids, cache) in enumerate(zip(sequences, caches, strict=True)):
            single = model.prefill(input_ids=mx.array([ids]), kv_cache=cache, compute=compute)
            for name in ("logits", "last_hidden_state"):
                np.testing.assert_allclose(
                    np.asarray(getattr(batch, name))[row, -len(ids):],
                    np.asarray(getattr(single, name))[0], atol=3e-6, rtol=3e-6,
                )
        for step in range(3):
            tokens = mx.array([[8 + step], [12 + step]])
            validity = mx.concatenate([mask, mx.ones((2, step + 1), dtype=mx.int32)], axis=1)
            batch = model.decode_step(
                input_ids=tokens, attention_mask=validity,
                position_ids=mx.array([[2 + step], [5 + step]]),
                kv_cache=batch_cache, compute=compute,
            )
            for row, cache in enumerate(caches):
                single = model.decode_step(input_ids=tokens[row:row + 1], kv_cache=cache, compute=compute)
                for name in ("logits", "last_hidden_state"):
                    np.testing.assert_allclose(
                        np.asarray(getattr(batch, name))[row], np.asarray(getattr(single, name))[0],
                        atol=3e-6, rtol=3e-6,
                    )


def test_cache_row_selection_preserves_live_prefix_when_growing():
    config = _tiny_config()
    cache = Qwen3ASRTextKVCache.allocate(config, batch_size=3, max_length=4)
    keys = mx.arange(3 * 2 * 3 * 4, dtype=mx.float32).reshape(3, 2, 3, 4)
    for layer in cache.layers:
        layer.append(keys, -keys)
    cache.prompt_length = 3
    cache.select_rows(mx.array([2, 0], dtype=mx.int32))
    cache.reserve(8)
    assert cache.current_length == cache.prompt_length == 3
    assert cache.max_length == 8
    for layer in cache.layers:
        assert layer.batch_size == 2
        assert layer.keys.shape == layer.values.shape == (2, 2, 8, 4)
        np.testing.assert_array_equal(np.asarray(layer.get()[0]), np.asarray(keys)[[2, 0]])
        np.testing.assert_array_equal(np.asarray(layer.get()[1]), -np.asarray(keys)[[2, 0]])
        layer.append(mx.ones((2, 2, 1, 4)), mx.zeros((2, 2, 1, 4)))
    assert cache.current_length == 4
