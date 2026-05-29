from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from mlx_speech.models.granite_speech_asr.config import GraniteSpeechTextConfig
from mlx_speech.models.granite_speech_asr.language_model import (
    GraniteCausalLM,
    greedy_next_token,
)


def _tiny_config(**overrides) -> GraniteSpeechTextConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_multiplier": 0.25,
        "embedding_multiplier": 3.0,
        "residual_multiplier": 0.5,
        "logits_scaling": 2.0,
        "mlp_bias": False,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    values.update(overrides)
    return GraniteSpeechTextConfig(**values)


def test_granite_lm_prefill_accepts_input_embeddings_and_cache():
    model = GraniteCausalLM(_tiny_config())
    inputs_embeds = mx.ones((1, 4, 16), dtype=mx.float32)

    out = model.prefill(inputs_embeds=inputs_embeds, max_cache_len=8)
    mx.eval(out.logits)

    assert out.logits.shape == (1, 4, 32)
    assert out.kv_cache is not None
    assert out.kv_cache.current_length == 4
    assert out.kv_cache.prompt_length == 4


def test_granite_lm_one_step_cached_decode_extends_cache():
    model = GraniteCausalLM(_tiny_config())
    out = model.prefill(input_ids=mx.array([[1, 2, 3]], dtype=mx.int32), max_cache_len=8)

    step = model.decode_step(
        input_ids=mx.array([[4]], dtype=mx.int32),
        kv_cache=out.kv_cache,
    )
    mx.eval(step.logits)

    assert step.logits.shape == (1, 1, 32)
    assert step.kv_cache.current_length == 4


def test_granite_lm_greedy_next_token_uses_final_position():
    logits = mx.array([[[0.0, 5.0, 1.0], [3.0, 1.0, 9.0]]], dtype=mx.float32)

    token = greedy_next_token(logits)

    np.testing.assert_array_equal(np.array(token), np.array([2]))


def test_granite_lm_parameter_names_match_checkpoint_subtree():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = GraniteCausalLM(_tiny_config())

    params = tree_flatten(Wrapper().parameters(), destination={})
    expected = {
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
    }

    assert expected <= set(params)


def test_granite_lm_stores_multiplier_config_on_runtime_modules():
    cfg = _tiny_config(
        embedding_multiplier=7.0,
        attention_multiplier=0.125,
        residual_multiplier=0.75,
        logits_scaling=4.0,
    )
    model = GraniteCausalLM(cfg)

    assert model.model.embedding_multiplier == 7.0
    assert model.model.layers[0].self_attn.scale == 0.125
    assert model.model.layers[0].residual_multiplier == 0.75
    assert model.logits_scaling == 4.0
