from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from mlx_speech.models.qwen3_asr.config import Qwen3ASRTextConfig
from mlx_speech.models.qwen3_asr.text_decoder import (
    Qwen3ASRTextAttention,
    Qwen3ASRTextForCausalLM,
    Qwen3ASRTextKVCache,
    Qwen3ASRTextModel,
    Qwen3ASRTextRMSNorm,
    Qwen3ASRTextRotaryEmbedding,
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
