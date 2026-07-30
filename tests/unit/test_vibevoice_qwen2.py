from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models._qwen2 import Qwen2Model as SharedQwen2Model
from mlx_speech.models.vibevoice.config import Qwen2LanguageConfig
from mlx_speech.models.vibevoice.qwen2 import (
    Qwen2Attention,
    Qwen2Model,
    Qwen2Output,
)


def _tiny_config() -> Qwen2LanguageConfig:
    return Qwen2LanguageConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=24,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        model_type="qwen2",
        tie_word_embeddings=False,
        rope_theta=1_000_000.0,
    )


def _set_deterministic_weights(module, *, dtype: mx.Dtype | None = None) -> None:
    weights = []
    parameters = tree_flatten(module.parameters(), destination={})
    for index, (name, parameter) in enumerate(parameters.items()):
        values = mx.arange(parameter.size, dtype=mx.float32).reshape(parameter.shape)
        values = mx.cos(values + float(index + 1)) * 0.03
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            values = values + 1.0
        weights.append((name, values.astype(parameter.dtype if dtype is None else dtype)))
    module.load_weights(weights, strict=True)


def test_vibevoice_qwen2_compatibility_exports_shared_model() -> None:
    assert Qwen2Model is SharedQwen2Model

    model = Qwen2Model(_tiny_config())
    _set_deterministic_weights(model)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    inputs_embeds = model.embed_tokens(input_ids)
    output = model(inputs_embeds=inputs_embeds)
    mx.eval(output.last_hidden_state)

    assert isinstance(output, Qwen2Output)
    assert output.last_hidden_state.shape == (1, 3, 16)
    assert output.past_key_values is output.cache
    assert isinstance(model.layers[0].self_attn, Qwen2Attention)
    assert model.layers[0].self_attn.num_heads == 4
    assert model.layers[0].self_attn.num_kv_heads == 2


def test_vibevoice_qwen2_parameter_surface_and_cached_call_are_preserved() -> None:
    model = Qwen2Model(_tiny_config())
    _set_deterministic_weights(model)
    names = set(tree_flatten(model.parameters(), destination={}))
    expected = {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.bias",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
    }
    assert expected <= names

    ids = mx.array([[4, 5, 6]], dtype=mx.int32)
    prefill = model(inputs_embeds=model.embed_tokens(ids[:, :2]))
    decode = model(
        inputs_embeds=model.embed_tokens(ids[:, 2:]),
        cache=prefill.cache,
    )
    mx.eval(decode.last_hidden_state)

    assert decode.last_hidden_state.shape == (1, 1, 16)
    assert all(keys.shape == (1, 3, 2, 4) for keys, _ in decode.cache)


def test_vibevoice_qwen2_preserves_legacy_bf16_rope_numerics() -> None:
    model = Qwen2Model(_tiny_config())
    _set_deterministic_weights(model, dtype=mx.bfloat16)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    output = model(inputs_embeds=model.embed_tokens(input_ids))
    actual = output.last_hidden_state[0, -1, :8]
    expected = mx.array(
        [
            -0.88868803,
            -1.48469543,
            -0.62630200,
            0.53879106,
            1.63290703,
            0.85080010,
            -0.22534780,
            -1.48673844,
        ],
        dtype=mx.float32,
    )
    mx.eval(actual)

    assert model.rotary_dtype_policy == "float32"
    assert output.last_hidden_state.dtype == mx.float32
    assert all(keys.dtype == mx.float32 for keys, _ in output.cache)
    assert output.cache[0][1].dtype == mx.bfloat16
    assert all(values.dtype == mx.float32 for _, values in output.cache[1:])
    assert mx.allclose(actual, expected, rtol=5e-4, atol=5e-4).item()
