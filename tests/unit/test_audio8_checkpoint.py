from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.audio8_asr_infinite.checkpoint import (
    expected_audio8_tensors,
    load_audio8_checkpoint,
    sanitize_state_dict,
    save_audio8_bf16_checkpoint,
)
from mlx_speech.models.audio8_asr_infinite.config import Audio8Config


def _config_payload() -> dict:
    return {
        "model_type": "audio8_asr_infinite",
        "weight_format_version": 2,
        "audio_config": {
            "model_type": "voxtral_realtime_encoder",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_dim": 4,
            "num_mel_bins": 4,
            "max_position_embeddings": 32,
            "sliding_window": 8,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 1000000.0},
            "streaming_n_left_pad_tokens": 4,
        },
        "text_config": {
            "model_type": "qwen2",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000.0},
            "tie_word_embeddings": True,
        },
        "projection_size": 32,
        "max_frame_len": 4,
        "supported_frame_lens": [2, 4],
        "audio_length_per_tok": 4,
        "audio_tower_frame_ms": 20,
        "streaming_n_left_pad_tokens": 4,
        "streaming_n_left_pad_tokens_by_frame_len": {"2": 6, "4": 3},
        "num_delay_tokens_by_frame_len": {"2": {"80": 2}, "4": {"160": 2}},
        "semantic_vad_horizons_seconds": [0.5, 1.0],
        "semantic_vad_num_classes": 8,
        "use_frame_len_embedding": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "auto_map": {"AutoConfig": "remote.Configuration"},
    }


def _source_weights(config: Audio8Config) -> dict[str, mx.array]:
    weights = {}
    for key, shape in expected_audio8_tensors(config).items():
        source_shape = (shape[0], shape[2], shape[1]) if key.endswith("conv1.weight") or key.endswith("conv2.weight") else shape
        weights[key] = mx.ones(source_shape, dtype=mx.bfloat16)
    return weights


def test_config_resolves_audio_clocks_and_delay_conditioning():
    config = Audio8Config.from_dict(_config_payload())

    assert config.clock_ms(2) == 40
    assert config.clock_ms(4) == 80
    assert config.left_pad_tokens(2) == 6
    assert config.delay_tokens(4, 160) == 2
    assert config.is_optimized_operating_point(4, 160)
    assert not config.is_optimized_operating_point(4, 80)
    assert "auto_map" not in config.to_dict()
    with pytest.raises(ValueError, match="multiple"):
        config.delay_tokens(4, 100)


def test_sanitize_maps_conv1d_layout_and_preserves_bf16():
    config = Audio8Config.from_dict(_config_payload())
    source = _source_weights(config)

    state, transposed = sanitize_state_dict(source, config)

    assert len(state) == 41
    assert set(transposed) == {
        "audio_tower.embedder.conv1.weight",
        "audio_tower.embedder.conv2.weight",
    }
    assert state["audio_tower.embedder.conv1.weight"].shape == (8, 3, 4)
    assert state["audio_tower.embedder.conv2.weight"].shape == (8, 3, 8)
    assert all(value.dtype == mx.bfloat16 for value in state.values())
    assert "language_model.lm_head.weight" not in state
    sanitize_state_dict(state, config)


def test_sanitize_rejects_schema_and_shape_differences():
    config = Audio8Config.from_dict(_config_payload())
    source = _source_weights(config)
    source.pop("audio_tower.norm.weight")
    source["unexpected.weight"] = mx.ones((1,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="1 missing, 1 unexpected"):
        sanitize_state_dict(source, config)

    source = _source_weights(config)
    source["language_model.model.norm.weight"] = mx.ones((7,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="shape mismatch"):
        sanitize_state_dict(source, config)


def test_conversion_merges_vad_and_copies_only_runtime_assets(tmp_path: Path):
    input_dir = tmp_path / "original"
    output_dir = tmp_path / "mlx-bf16"
    input_dir.mkdir()
    config = Audio8Config.from_dict(_config_payload())
    (input_dir / "config.json").write_text(json.dumps(_config_payload()), encoding="utf-8")
    (input_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (input_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (input_dir / "modeling_audio8_asr_infinite.py").write_text("remote", encoding="utf-8")
    source = _source_weights(config)
    vad_keys = [key for key in source if key.startswith("semantic_vad_heads.")]
    mx.save_safetensors(
        input_dir / "model.safetensors",
        {key: value for key, value in source.items() if key not in vad_keys},
    )
    mx.save_safetensors(
        input_dir / "semantic_vad_heads.safetensors",
        {key: source[key] for key in vad_keys},
    )

    checkpoint = load_audio8_checkpoint(input_dir)
    report = save_audio8_bf16_checkpoint(checkpoint, output_dir)
    reloaded = load_audio8_checkpoint(output_dir)

    assert report.tensor_count == len(source) == len(reloaded.state_dict)
    assert reloaded.transposed_keys == ()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "preprocessor_config.json").exists()
    assert not (output_dir / "modeling_audio8_asr_infinite.py").exists()
    assert not (output_dir / "semantic_vad_heads.safetensors").exists()
    assert "auto_map" not in json.loads((output_dir / "config.json").read_text())
