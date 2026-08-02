from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from mlx_speech.models._cache import BoundedKVCache
from mlx_speech.models.dots_tts.config import DotsTTSQwenConfig
from mlx_speech.models.dots_tts.qwen import DotsTTSQwen


def _tiny_config(**overrides) -> DotsTTSQwenConfig:
    values = {
        "vocab_size": 24,
        "max_position_embeddings": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "model_type": "qwen2",
        "extra": {
            "use_cache": True,
            "use_sliding_window": False,
            "rope_scaling": None,
        },
    }
    values.update(overrides)
    return DotsTTSQwenConfig(**values)


def _set_deterministic_weights(module, *, dtype: mx.Dtype | None = None) -> None:
    weights = []
    parameters = tree_flatten(module.parameters(), destination={})
    for index, (name, parameter) in enumerate(parameters.items()):
        values = mx.arange(parameter.size, dtype=mx.float32).reshape(parameter.shape)
        values = mx.sin(values + float(index + 1)) * 0.04
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            values = values + 1.0
        weights.append(
            (name, values.astype(parameter.dtype if dtype is None else dtype))
        )
    module.load_weights(weights, strict=True)


def test_dots_qwen_ids_embeddings_gqa_tied_logits_and_eos() -> None:
    model = DotsTTSQwen(_tiny_config())
    _set_deterministic_weights(model)
    input_ids = mx.array([[1, 3, 5, 7]], dtype=mx.int32)

    from_ids = model(input_ids=input_ids)
    input_embeddings = model.get_input_embeddings()(input_ids)
    from_embeddings = model(inputs_embeds=input_embeddings)
    expected_logits = model.get_output_embeddings().as_linear(
        from_ids.last_hidden_state
    )
    probabilities = model.eos_probabilities(from_ids.last_hidden_state)
    stop = model.should_stop(from_ids.last_hidden_state, threshold=0.0)
    mx.eval(
        from_ids.last_hidden_state,
        from_embeddings.last_hidden_state,
        from_ids.logits,
        expected_logits,
        from_ids.eos_logits,
        probabilities,
        stop,
    )

    assert from_ids.last_hidden_state.shape == (1, 4, 16)
    assert mx.allclose(
        from_ids.last_hidden_state,
        from_embeddings.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
    ).item()
    assert model.get_output_embeddings() is model.get_input_embeddings()
    assert from_ids.logits.shape == (1, 4, 24)
    assert mx.allclose(from_ids.logits, expected_logits).item()
    assert from_ids.eos_logits.shape == (1, 4, 2)
    assert probabilities.shape == (1, 4)
    assert mx.all(stop).item()
    assert len(from_ids.cache) == 2
    assert all(isinstance(cache, BoundedKVCache) for cache in from_ids.cache)
    assert all(cache.offset == 4 for cache in from_ids.cache)
    assert all(cache.capacity == 32 for cache in from_ids.cache)
    assert all(keys.shape == (1, 4, 2, 4) for keys, _ in from_ids.cache)
    assert all(values.shape == (1, 4, 2, 4) for _, values in from_ids.cache)


def test_dots_qwen_can_skip_eos_projection(monkeypatch) -> None:
    model = DotsTTSQwen(_tiny_config())
    _set_deterministic_weights(model)

    def unexpected_eos(_hidden_states):
        raise AssertionError("disabled EOS built its projection")

    monkeypatch.setattr(model, "eos_logits", unexpected_eos)
    output = model.step(
        input_ids=mx.array([[1, 2]], dtype=mx.int32),
        request_eos=False,
    )
    mx.eval(output.last_hidden_state)
    assert output.eos_logits is None


def test_dots_qwen_full_and_incremental_decode_agree() -> None:
    model = DotsTTSQwen(_tiny_config())
    _set_deterministic_weights(model)
    input_ids = mx.array([[2, 4, 6, 8, 10]], dtype=mx.int32)

    full = model.step(input_ids=input_ids)
    cache = None
    decoded = []
    storage_ids = None
    for index in range(input_ids.shape[1]):
        step = model.step(input_ids=input_ids[:, index : index + 1], cache=cache)
        decoded.append(step.last_hidden_state)
        cache = step.cache
        if storage_ids is None:
            storage_ids = tuple(
                (id(layer_cache.keys), id(layer_cache.values)) for layer_cache in cache
            )
        else:
            assert (
                tuple(
                    (id(layer_cache.keys), id(layer_cache.values))
                    for layer_cache in cache
                )
                == storage_ids
            )
        assert all(keys.shape[1] == index + 1 for keys, _ in cache)
    incremental = mx.concatenate(decoded, axis=1)
    mx.eval(full.last_hidden_state, incremental)

    assert full.logits is None
    assert not hasattr(model, "sampler")
    assert mx.allclose(
        full.last_hidden_state,
        incremental,
        rtol=2e-3,
        atol=2e-3,
    ).item()


def test_dots_qwen_keeps_query_dtype_rope_for_bf16() -> None:
    model = DotsTTSQwen(_tiny_config())
    _set_deterministic_weights(model, dtype=mx.bfloat16)

    output = model.step(input_ids=mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(output.last_hidden_state)

    assert model.model.rotary_dtype_policy == "query"
    assert output.last_hidden_state.dtype == mx.bfloat16
    assert all(keys.dtype == mx.bfloat16 for keys, _ in output.cache)
    assert all(values.dtype == mx.bfloat16 for _, values in output.cache)


def test_dots_qwen_builds_rotary_geometry_once_for_all_layers(monkeypatch) -> None:
    model = DotsTTSQwen(_tiny_config(num_hidden_layers=3))
    calls = 0
    original = model.model.rotary_emb

    def count_rotary(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.model, "rotary_emb", count_rotary)
    output = model.step(input_ids=mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(output.last_hidden_state)

    assert calls == 1


def test_dots_qwen_exact_cache_capacity_is_slice_written_and_bounded() -> None:
    model = DotsTTSQwen(_tiny_config(max_position_embeddings=16))
    _set_deterministic_weights(model)

    prefill = model.step(
        input_ids=mx.array([[1, 2, 3]], dtype=mx.int32),
        cache_capacity=5,
    )
    storage_ids = tuple(
        (id(layer_cache.keys), id(layer_cache.values)) for layer_cache in prefill.cache
    )
    decode = model.step(
        input_ids=mx.array([[4, 5]], dtype=mx.int32),
        cache=prefill.cache,
        cache_capacity=5,
    )
    mx.eval(decode.last_hidden_state)

    assert all(layer_cache.offset == 5 for layer_cache in decode.cache)
    assert all(layer_cache.capacity == 5 for layer_cache in decode.cache)
    assert (
        tuple(
            (id(layer_cache.keys), id(layer_cache.values))
            for layer_cache in decode.cache
        )
        == storage_ids
    )
    with pytest.raises(ValueError, match="cache_capacity"):
        model.step(
            input_ids=mx.array([[6]], dtype=mx.int32),
            cache=decode.cache,
            cache_capacity=5,
        )


def test_dots_qwen_default_cache_grows_in_256_token_blocks() -> None:
    model = DotsTTSQwen(_tiny_config(max_position_embeddings=600, num_hidden_layers=1))
    _set_deterministic_weights(model)
    prefill = model.step(input_ids=mx.zeros((1, 255), dtype=mx.int32))
    assert prefill.cache[0].capacity == 256

    decode = model.step(
        input_ids=mx.zeros((1, 2), dtype=mx.int32),
        cache=prefill.cache,
    )
    mx.eval(decode.last_hidden_state)

    assert decode.cache[0].offset == 257
    assert decode.cache[0].capacity == 512
    assert decode.cache[0].max_capacity == 600


def test_dots_qwen_rejected_later_layer_append_restores_all_offsets() -> None:
    model = DotsTTSQwen(_tiny_config())
    _set_deterministic_weights(model)
    prefill = model.step(input_ids=mx.array([[1, 2]], dtype=mx.int32))
    prior_offsets = tuple(cache.offset for cache in prefill.cache)

    later_cache = prefill.cache[-1]
    later_cache.values = later_cache.values.astype(mx.bfloat16)
    with pytest.raises(ValueError, match="dtypes differ"):
        model.step(
            input_ids=mx.array([[3]], dtype=mx.int32),
            cache=prefill.cache,
        )

    assert tuple(cache.offset for cache in prefill.cache) == prior_offsets
    assert all(cache.fetch()[0].shape[1] == 2 for cache in prefill.cache)
    assert all(cache.fetch()[1].shape[1] == 2 for cache in prefill.cache)


def test_dots_qwen_rejects_invalid_inputs_cache_and_eos_threshold() -> None:
    model = DotsTTSQwen(_tiny_config(max_position_embeddings=2))
    input_ids = mx.array([[1, 2]], dtype=mx.int32)
    embeddings = mx.zeros((1, 2, 16), dtype=mx.float32)

    with pytest.raises(ValueError, match="exactly one"):
        model()
    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=input_ids, inputs_embeds=embeddings)
    with pytest.raises(ValueError, match="input_ids"):
        model(input_ids=mx.array([1, 2], dtype=mx.int32))
    with pytest.raises(ValueError, match="embedding width"):
        model(inputs_embeds=mx.zeros((1, 2, 15), dtype=mx.float32))
    with pytest.raises(ValueError, match="cache layer count"):
        model(input_ids=input_ids[:, :1], cache=[])
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(input_ids=mx.array([[1, 2, 3]], dtype=mx.int32))
    with pytest.raises(ValueError, match="cache_capacity must be positive"):
        model(input_ids=input_ids[:, :1], cache_capacity=0)
    with pytest.raises(ValueError, match="cache_capacity exceeds"):
        model(input_ids=input_ids[:, :1], cache_capacity=3)
    with pytest.raises(ValueError, match="threshold"):
        model.should_stop(embeddings, threshold=1.1)
