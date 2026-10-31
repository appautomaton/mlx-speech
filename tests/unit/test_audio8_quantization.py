from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_speech.models.audio8_asr_infinite.checkpoint import (
    Audio8Checkpoint,
    Audio8QuantizationConfig,
    get_audio8_quantization,
    load_audio8_checkpoint,
    load_checkpoint_into_model,
    quantize_audio8_model,
    sanitize_state_dict,
    save_audio8_model,
)
from mlx_speech.models.audio8_asr_infinite.config import Audio8Config
from mlx_speech.models.audio8_asr_infinite.model import Audio8ASRModel
from tests.unit.test_audio8_checkpoint import _config_payload, _source_weights


def _quantization_payload():
    payload = _config_payload()
    payload["audio_config"].update(
        hidden_size=32, intermediate_size=64, num_mel_bins=32
    )
    payload["text_config"].update(hidden_size=32, intermediate_size=64)
    payload["projection_size"] = 128
    return payload


def _quantized_model():
    config = Audio8Config.from_dict(_quantization_payload())
    model = Audio8ASRModel(config)
    state, _ = sanitize_state_dict(_source_weights(config), config)
    load_checkpoint_into_model(
        model,
        Audio8Checkpoint(Path("."), config, state, (), ()),
        strict=True,
    )
    quantization = Audio8QuantizationConfig(bits=8, group_size=32, mode="affine")
    quantize_audio8_model(model, quantization)
    return model, quantization


def test_affine_int8_save_reload_restores_exact_quantized_layout(tmp_path: Path):
    model, quantization = _quantized_model()
    inputs = mx.random.normal((1, 3, 32))
    t_cond = mx.random.normal((1, 1, 32)) * 0.1
    expected = model.language_model(
        inputs_embeds=inputs, t_cond=t_cond, logits_to_keep=0
    ).logits

    save_audio8_model(
        model,
        tmp_path,
        quantization=quantization,
        copy_supporting_from=None,
    )
    checkpoint = load_audio8_checkpoint(tmp_path)
    restored = Audio8ASRModel(checkpoint.config)
    report = load_checkpoint_into_model(restored, checkpoint, strict=True)
    actual = restored.language_model(
        inputs_embeds=inputs, t_cond=t_cond, logits_to_keep=0
    ).logits
    mx.eval(expected, actual)

    assert get_audio8_quantization(checkpoint.config) == quantization
    assert report.checkpoint_only == ()
    assert report.shape_mismatches == ()
    assert set(report.model_only) == {
        "audio_tower.rotary_emb.inv_freq",
        "time_embedding.inv_freq",
    }
    assert restored.language_model.lm_head is None
    assert isinstance(restored.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert restored.audio_tower.embedder.conv1.weight.shape == (32, 3, 32)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_int8_preserves_sensitive_projections_in_bf16():
    model, _ = _quantized_model()
    protected = [
        model.multi_modal_projector.linear_1,
        model.multi_modal_projector.linear_2,
    ]
    protected.extend(model.semantic_vad_heads)
    for layer in model.language_model.model.layers:
        protected.extend((layer.ada_rms_norm.linear1, layer.ada_rms_norm.linear2))

    assert all(type(module) is nn.Linear for module in protected)
    assert all(module.weight.dtype == mx.bfloat16 for module in protected)
    assert type(model.frame_len_embedding) is nn.Embedding
    assert model.frame_len_embedding.weight.dtype == mx.bfloat16
    assert isinstance(model.audio_tower.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(
        model.language_model.model.layers[0].mlp.down_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)


def test_fp32_condition_does_not_promote_quantized_decoder():
    model, _ = _quantized_model()
    inputs = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
    t_cond = mx.random.normal((1, 1, 32), dtype=mx.float32) * 0.1

    output = model.language_model(inputs_embeds=inputs, t_cond=t_cond)
    mx.eval(output.logits, output.last_hidden_state)

    assert output.logits.dtype == mx.bfloat16
    assert output.last_hidden_state.dtype == mx.bfloat16
    assert all(layer.keys.dtype == mx.bfloat16 for layer in output.cache)
    assert all(layer.values.dtype == mx.bfloat16 for layer in output.cache)


def test_int8_package_records_config_and_excludes_generated_buffers(tmp_path: Path):
    model, quantization = _quantized_model()
    save_audio8_model(
        model,
        tmp_path,
        quantization=quantization,
        copy_supporting_from=None,
    )

    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    weights = mx.load(str(tmp_path / "model.safetensors"))

    assert payload["quantization"] == quantization.to_dict()
    assert not any(key.startswith("language_model.lm_head.") for key in weights)
    assert weights["language_model.model.embed_tokens.weight"].dtype == mx.uint32
    assert weights["language_model.model.embed_tokens.scales"].dtype == mx.bfloat16
    assert weights["language_model.model.embed_tokens.biases"].dtype == mx.bfloat16
    assert weights["multi_modal_projector.linear_1.weight"].dtype == mx.bfloat16
    assert "multi_modal_projector.linear_1.scales" not in weights
    assert "frame_len_embedding.scales" not in weights
    assert "audio_tower.rotary_emb.inv_freq" not in weights
    assert "time_embedding.inv_freq" not in weights


def test_legacy_duplicate_tied_weights_load_only_when_identical(tmp_path: Path):
    model, quantization = _quantized_model()
    save_audio8_model(
        model,
        tmp_path,
        quantization=quantization,
        copy_supporting_from=None,
    )
    checkpoint = load_audio8_checkpoint(tmp_path)
    state = dict(checkpoint.state_dict)
    for suffix in ("weight", "scales", "biases"):
        state[f"language_model.lm_head.{suffix}"] = state[
            f"language_model.model.embed_tokens.{suffix}"
        ]
    legacy = Audio8Checkpoint(
        checkpoint.model_dir,
        checkpoint.config,
        state,
        checkpoint.source_files,
        checkpoint.transposed_keys,
    )
    restored = Audio8ASRModel(legacy.config)

    report = load_checkpoint_into_model(restored, legacy, strict=True)

    assert report.checkpoint_only == ()
    assert restored.language_model.lm_head is None
    assert "language_model.lm_head.weight" in legacy.state_dict
    state["language_model.lm_head.scales"] = (
        state["language_model.lm_head.scales"] + 1
    )
    with pytest.raises(ValueError, match="tied weights differ"):
        load_checkpoint_into_model(Audio8ASRModel(legacy.config), legacy, strict=True)
