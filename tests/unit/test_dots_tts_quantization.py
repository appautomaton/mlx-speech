from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.checkpoint import (
    BASE_DTYPE_POLICY,
    SOURCE_REVISIONS,
    DotsTTSArtifactConfig,
    DotsTTSCoreComponents,
    DotsTTSQuantizationConfig,
    artifact_tensor_dtype,
    eligible_qwen_quantization_paths,
    quantize_dots_tts_core,
)
from mlx_speech.models.dots_tts.config import DotsTTSConfig, DotsTTSQwenConfig
from tests.unit.test_dots_tts_config import dots_config


def _core() -> DotsTTSCoreComponents:
    qwen = DotsTTSQwenConfig.from_dict(
        {
            "model_type": "qwen2",
            "vocab_size": 128,
            "max_position_embeddings": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": True,
            "hidden_act": "silu",
        }
    )
    config_payload = dots_config()
    config_payload["DiT"]["modulation"] = True
    core = DotsTTSCoreComponents(DotsTTSConfig.from_dict(config_payload), qwen)
    core.set_dtype(mx.bfloat16)
    return core


def _quantization(paths: tuple[str, ...]) -> DotsTTSQuantizationConfig:
    return DotsTTSQuantizationConfig(
        bits=8,
        group_size=64,
        mode="affine",
        module_types=("Linear", "Embedding"),
        path_prefixes=("qwen.model.",),
        quantized_paths=paths,
    )


def _artifact_config(
    quantization: DotsTTSQuantizationConfig,
) -> DotsTTSArtifactConfig:
    source = SOURCE_REVISIONS["soar"]
    return DotsTTSArtifactConfig.from_dict(
        {
            "schema_version": 1,
            "model_family": "dots_tts",
            "variant": "soar",
            "mode": "flow_matching",
            "artifact_class": "int8",
            "source": {**source, "manifest_sha256": "a" * 64},
            "dtype_policy": BASE_DTYPE_POLICY,
            "quantization": quantization.to_dict(),
        }
    )


def test_quantization_selects_every_eligible_native_qwen_module_only() -> None:
    core = _core()
    paths = eligible_qwen_quantization_paths(core)
    assert paths == (
        "qwen.model.embed_tokens",
        "qwen.model.layers.0.mlp.down_proj",
        "qwen.model.layers.0.mlp.gate_proj",
        "qwen.model.layers.0.mlp.up_proj",
        "qwen.model.layers.0.self_attn.k_proj",
        "qwen.model.layers.0.self_attn.o_proj",
        "qwen.model.layers.0.self_attn.q_proj",
        "qwen.model.layers.0.self_attn.v_proj",
    )

    quantization = _quantization(paths)
    quantize_dots_tts_core(core, quantization)
    modules = dict(core.named_modules())
    assert all(
        isinstance(modules[path], (nn.QuantizedLinear, nn.QuantizedEmbedding))
        for path in paths
    )
    assert isinstance(modules["qwen.eos_proj.linear1"], nn.Linear)
    assert isinstance(modules["hidden_projection"], nn.Linear)

    parameters = tree_flatten(core.parameters(), destination={})
    for path in paths:
        assert parameters[f"{path}.weight"].dtype == mx.uint32
        assert parameters[f"{path}.scales"].dtype == mx.bfloat16
        assert parameters[f"{path}.biases"].dtype == mx.bfloat16
    assert parameters["qwen.eos_proj.linear1.weight"].dtype == mx.bfloat16

    output = core.qwen(input_ids=mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(output.last_hidden_state, output.eos_logits, output.logits)
    assert output.last_hidden_state.dtype == mx.bfloat16
    assert output.eos_logits.dtype == mx.bfloat16
    assert output.logits is not None and output.logits.dtype == mx.bfloat16


def test_quantized_qwen_fuses_decode_projections_after_loading() -> None:
    core = _core()
    paths = eligible_qwen_quantization_paths(core)
    quantize_dots_tts_core(core, _quantization(paths))
    layer = core.qwen.model.layers[0]
    value = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    expected_qkv = (
        layer.self_attn.q_proj(value),
        layer.self_attn.k_proj(value),
        layer.self_attn.v_proj(value),
    )
    expected_gate_up = (
        layer.mlp.gate_proj(value),
        layer.mlp.up_proj(value),
    )
    mx.eval(expected_qkv, expected_gate_up)

    core.qwen.model.fuse_for_inference()
    actual_qkv = layer.self_attn.qkv_proj.split(value)
    actual_gate_up = layer.mlp.gate_up_proj.split(value)
    mx.eval(actual_qkv, actual_gate_up)
    for actual, expected in zip(
        (*actual_qkv, *actual_gate_up),
        (*expected_qkv, *expected_gate_up),
        strict=True,
    ):
        assert float(mx.max(mx.abs(actual - expected)).item()) <= 0.02

    parameters = tree_flatten(core.parameters(), destination={})
    assert parameters["qwen.model.layers.0.self_attn.qkv_proj.weight"].dtype == mx.uint32
    assert parameters["qwen.model.layers.0.mlp.gate_up_proj.weight"].dtype == mx.uint32
    assert not any("layers.0.self_attn.q_proj" in name for name in parameters)
    assert not any("layers.0.mlp.gate_proj" in name for name in parameters)
    core.qwen.model.fuse_for_inference()


def test_base_qwen_fuses_decode_projections_without_quantization() -> None:
    core = _core()
    input_ids = mx.array([[1, 2]], dtype=mx.int32)
    expected = core.qwen(input_ids=input_ids)
    mx.eval(expected.last_hidden_state, expected.eos_logits, expected.logits)

    core.qwen.model.fuse_for_inference()
    actual = core.qwen(input_ids=input_ids)
    mx.eval(actual.last_hidden_state, actual.eos_logits, actual.logits)
    assert (
        float(
            mx.max(
                mx.abs(actual.last_hidden_state - expected.last_hidden_state)
            ).item()
        )
        <= 0.02
    )
    assert float(mx.max(mx.abs(actual.eos_logits - expected.eos_logits)).item()) <= 0.02
    assert float(mx.max(mx.abs(actual.logits - expected.logits)).item()) <= 0.02

    parameters = tree_flatten(core.parameters(), destination={})
    assert (
        parameters["qwen.model.layers.0.self_attn.qkv_proj.weight"].dtype
        == mx.bfloat16
    )
    assert (
        parameters["qwen.model.layers.0.mlp.gate_up_proj.weight"].dtype
        == mx.bfloat16
    )


def test_quantization_metadata_reconstructs_an_exact_complete_predicate() -> None:
    core = _core()
    paths = eligible_qwen_quantization_paths(core)
    with pytest.raises(ValueError, match="predicate differs"):
        quantize_dots_tts_core(core, _quantization(paths[:-1]))

    payload = _quantization(paths).to_dict()
    payload["path_prefixes"] = ["llm."]
    with pytest.raises(ValueError, match=r"qwen\.model"):
        DotsTTSQuantizationConfig.from_dict(payload)

    payload = _quantization(paths).to_dict()
    payload["quantized_paths"] = list(reversed(paths))
    with pytest.raises(ValueError, match="unique and sorted"):
        DotsTTSQuantizationConfig.from_dict(payload)


def test_artifact_dtype_resolution_separates_packed_and_base_tensors() -> None:
    core = _core()
    paths = eligible_qwen_quantization_paths(core)
    artifact = _artifact_config(_quantization(paths))
    selected = paths[0]
    assert artifact_tensor_dtype(artifact, "core", f"{selected}.weight") == mx.uint32
    assert (
        artifact_tensor_dtype(artifact, "core", f"{selected}.scales")
        == mx.bfloat16
    )
    assert (
        artifact_tensor_dtype(artifact, "core", "qwen.eos_proj.linear1.weight")
        == mx.bfloat16
    )
    assert (
        artifact_tensor_dtype(artifact, "vocoder", "audio_encoder.pre_conv.weight")
        == mx.float32
    )
