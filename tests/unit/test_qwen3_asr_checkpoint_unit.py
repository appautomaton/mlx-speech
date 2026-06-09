from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from scripts.convert.qwen3_asr import convert_qwen3_asr
from mlx_speech.models.qwen3_asr.config import Qwen3ASRConfig
from mlx_speech.models.qwen3_asr.checkpoint import (
    Qwen3ASRCheckpoint,
    build_alignment_report,
    load_checkpoint_into_model,
    load_qwen3_asr_checkpoint,
    sanitize_key,
    sanitize_state_dict,
)


def _config_payload() -> dict:
    return {
        "model_type": "qwen3_asr",
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "model_type": "qwen3_asr",
            "audio_token_id": 10,
            "audio_start_token_id": 11,
            "audio_end_token_id": 12,
            "audio_config": {
                "d_model": 16,
                "num_mel_bins": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 32,
                "downsample_hidden_size": 4,
                "output_dim": 16,
                "max_source_positions": 64,
                "n_window": 50,
                "n_window_infer": 800,
                "conv_chunksize": 2,
            },
            "text_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "tie_word_embeddings": True,
            },
        },
    }


def _write_minimal_config(path: Path) -> None:
    (path / "config.json").write_text(json.dumps(_config_payload()), encoding="utf-8")


def test_qwen3_asr_sanitize_key_maps_upstream_namespaces():
    assert sanitize_key("thinker.audio_tower.conv2d1.weight") == "audio_tower.conv2d1.weight"
    assert (
        sanitize_key("thinker.model.layers.0.self_attn.q_norm.weight")
        == "text_decoder.model.layers.0.self_attn.q_norm.weight"
    )
    assert sanitize_key("thinker.lm_head.weight") == "text_decoder.lm_head.weight"
    assert sanitize_key("unrelated.weight") is None


def test_qwen3_asr_sanitizer_renames_and_transposes_conv2d_weights():
    conv = mx.array(np.arange(36, dtype=np.float32).reshape(4, 1, 3, 3))

    state, skipped, renamed, transposed = sanitize_state_dict(
        {
            "thinker.audio_tower.conv2d1.weight": conv,
            "thinker.model.embed_tokens.weight": mx.ones((8, 4), dtype=mx.bfloat16),
            "thinker.lm_head.weight": mx.ones((8, 4), dtype=mx.bfloat16),
            "unused.weight": mx.ones((1,)),
        }
    )

    assert skipped == ("unused.weight",)
    assert ("thinker.audio_tower.conv2d1.weight", "audio_tower.conv2d1.weight") in renamed
    assert transposed == ("audio_tower.conv2d1.weight",)
    assert state["audio_tower.conv2d1.weight"].shape == (4, 3, 3, 1)
    assert state["text_decoder.model.embed_tokens.weight"].dtype == mx.bfloat16
    assert state["text_decoder.lm_head.weight"].dtype == mx.bfloat16


def test_qwen3_asr_sanitizer_leaves_converted_conv2d_layout():
    state, _, _, transposed = sanitize_state_dict(
        {"audio_tower.conv2d1.weight": mx.ones((4, 3, 3, 1))}
    )

    assert transposed == ()
    assert state["audio_tower.conv2d1.weight"].shape == (4, 3, 3, 1)


def test_qwen3_asr_sanitizer_rejects_duplicate_sanitized_keys():
    with pytest.raises(ValueError, match="Duplicate"):
        sanitize_state_dict(
            {
                "thinker.lm_head.weight": mx.ones((1, 1)),
                "text_decoder.lm_head.weight": mx.ones((1, 1)),
            }
        )


def test_qwen3_asr_checkpoint_loader_supports_sharded_index(tmp_path):
    _write_minimal_config(tmp_path)
    mx.save_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "thinker.audio_tower.conv2d1.weight": mx.ones((4, 1, 3, 3)),
            "thinker.model.embed_tokens.weight": mx.ones((8, 4), dtype=mx.bfloat16),
        },
    )
    mx.save_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        {"thinker.lm_head.weight": mx.ones((8, 4), dtype=mx.bfloat16)},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {
                    "thinker.audio_tower.conv2d1.weight": "model-00001-of-00002.safetensors",
                    "thinker.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "thinker.lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoint = load_qwen3_asr_checkpoint(tmp_path)

    assert len(checkpoint.source_files) == 2
    assert checkpoint.config.model_type == "qwen3_asr"
    assert checkpoint.transposed_keys == ("audio_tower.conv2d1.weight",)
    assert checkpoint.state_dict["audio_tower.conv2d1.weight"].shape == (4, 3, 3, 1)


def test_qwen3_asr_conversion_writes_bf16_safetensors_and_supporting_files(tmp_path):
    input_dir = tmp_path / "original"
    output_dir = tmp_path / "mlx-bf16"
    input_dir.mkdir()
    _write_minimal_config(input_dir)
    (input_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    mx.save_safetensors(
        input_dir / "model.safetensors",
        {
            "thinker.audio_tower.conv2d1.weight": mx.ones((4, 1, 3, 3), dtype=mx.bfloat16),
            "thinker.model.embed_tokens.weight": mx.ones((8, 4), dtype=mx.bfloat16),
            "thinker.lm_head.weight": mx.ones((8, 4), dtype=mx.bfloat16),
        },
    )

    report = convert_qwen3_asr(input_dir, output_dir)
    converted = mx.load(str(output_dir / "model.safetensors"))

    assert report.tensor_count == 3
    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer_config.json").exists()
    assert converted["audio_tower.conv2d1.weight"].shape == (4, 3, 3, 1)
    assert converted["audio_tower.conv2d1.weight"].dtype == mx.bfloat16
    assert converted["text_decoder.model.embed_tokens.weight"].dtype == mx.bfloat16


def test_qwen3_asr_alignment_report_splits_missing_and_shape_mismatch():
    report = build_alignment_report(
        {
            "shared.weight": mx.ones((2, 3)),
            "model.only": mx.ones((1,)),
        },
        {
            "shared.weight": mx.ones((3, 2)),
            "checkpoint.only": mx.ones((1,)),
        },
    )

    assert report.checkpoint_only == ("checkpoint.only",)
    assert report.model_only == ("model.only",)
    assert report.shape_mismatches == (("shared.weight", (2, 3), (3, 2)),)
    assert not report.is_exact_match


def test_qwen3_asr_strict_load_allows_generated_audio_positions():
    class FakeModel:
        def __init__(self):
            self.loaded = None

        def parameters(self):
            return {
                "audio_tower": {
                    "positional_embedding": {
                        "positional_embedding": mx.ones((4, 4)),
                    },
                },
                "text_decoder": {
                    "model": {
                        "embed_tokens": {
                            "weight": mx.ones((2, 2)),
                        },
                    },
                },
            }

        def load_weights(self, weights, *, strict: bool = True):
            self.loaded = (tuple(weights), strict)

    model = FakeModel()
    checkpoint = Qwen3ASRCheckpoint(
        model_dir=Path("."),
        config=Qwen3ASRConfig.from_dict(_config_payload()),
        state_dict={"text_decoder.model.embed_tokens.weight": mx.ones((2, 2))},
        source_files=(),
        skipped_keys=(),
        renamed_keys=(),
        transposed_keys=(),
    )

    report = load_checkpoint_into_model(model, checkpoint, strict=True)

    assert report.model_only == ("audio_tower.positional_embedding.positional_embedding",)
    assert report.unexpected_model_only == ()
    assert not report.is_exact_match
    assert report.is_loadable_match
    assert model.loaded is not None
    assert model.loaded[1] is False


def test_qwen3_asr_strict_load_rejects_unexpected_model_only_key():
    class FakeModel:
        def parameters(self):
            return {
                "unexpected": {
                    "weight": mx.ones((1,)),
                },
            }

        def load_weights(self, weights, *, strict: bool = True):
            raise AssertionError("unexpected partial load")

    checkpoint = Qwen3ASRCheckpoint(
        model_dir=Path("."),
        config=Qwen3ASRConfig.from_dict(_config_payload()),
        state_dict={},
        source_files=(),
        skipped_keys=(),
        renamed_keys=(),
        transposed_keys=(),
    )

    with pytest.raises(ValueError, match="1 unexpected model-only"):
        load_checkpoint_into_model(FakeModel(), checkpoint, strict=True)
