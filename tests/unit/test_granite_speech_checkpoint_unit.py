from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from mlx_speech.models.granite_speech_asr.checkpoint import (
    build_alignment_report,
    load_granite_speech_checkpoint,
    sanitize_state_dict,
)


def _minimal_config() -> dict:
    return {
        "model_type": "granite_speech",
        "audio_token_index": 100352,
        "downsample_rate": 5,
        "window_size": 15,
        "encoder_config": {"input_dim": 160},
        "projector_config": {},
        "text_config": {},
    }


def test_granite_checkpoint_sanitizer_skips_batchnorm_bookkeeping():
    state, skipped, transposed = sanitize_state_dict(
        {
            "encoder.layers.0.conv.batch_norm.num_batches_tracked": mx.array(0),
            "encoder.input_linear.weight": mx.ones((2, 2)),
        }
    )

    assert "encoder.input_linear.weight" in state
    assert skipped == ("encoder.layers.0.conv.batch_norm.num_batches_tracked",)
    assert transposed == ()


def test_granite_checkpoint_sanitizer_transposes_original_conv1d_weights():
    up = mx.array(np.arange(12, dtype=np.float32).reshape(4, 3, 1))
    down = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    depth = mx.array(np.arange(60, dtype=np.float32).reshape(4, 1, 15))

    state, skipped, transposed = sanitize_state_dict(
        {
            "encoder.layers.0.conv.up_conv.weight": up,
            "encoder.layers.0.conv.down_conv.weight": down,
            "encoder.layers.0.conv.depth_conv.conv.weight": depth,
        }
    )

    assert skipped == ()
    assert transposed == (
        "encoder.layers.0.conv.up_conv.weight",
        "encoder.layers.0.conv.down_conv.weight",
        "encoder.layers.0.conv.depth_conv.conv.weight",
    )
    assert state["encoder.layers.0.conv.up_conv.weight"].shape == (4, 1, 3)
    assert state["encoder.layers.0.conv.down_conv.weight"].shape == (3, 1, 4)
    assert state["encoder.layers.0.conv.depth_conv.conv.weight"].shape == (4, 15, 1)


def test_granite_checkpoint_sanitizer_leaves_converted_conv1d_weights():
    state, _, transposed = sanitize_state_dict(
        {
            "encoder.layers.0.conv.up_conv.weight": mx.ones((4, 1, 3)),
            "encoder.layers.0.conv.depth_conv.conv.weight": mx.ones((4, 15, 1)),
        }
    )

    assert transposed == ()
    assert state["encoder.layers.0.conv.up_conv.weight"].shape == (4, 1, 3)
    assert state["encoder.layers.0.conv.depth_conv.conv.weight"].shape == (4, 15, 1)


def test_granite_checkpoint_loader_supports_sharded_index(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()), encoding="utf-8")
    mx.save_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "encoder.layers.0.conv.up_conv.weight": mx.ones((4, 3, 1)),
            "encoder.layers.0.conv.batch_norm.num_batches_tracked": mx.array(0),
        },
    )
    mx.save_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        {"language_model.model.embed_tokens.weight": mx.ones((2, 2))},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {
                    "encoder.layers.0.conv.up_conv.weight": "model-00001-of-00002.safetensors",
                    "encoder.layers.0.conv.batch_norm.num_batches_tracked": "model-00001-of-00002.safetensors",
                    "language_model.model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoint = load_granite_speech_checkpoint(tmp_path)

    assert len(checkpoint.source_files) == 2
    assert checkpoint.skipped_keys == ("encoder.layers.0.conv.batch_norm.num_batches_tracked",)
    assert checkpoint.transposed_keys == ("encoder.layers.0.conv.up_conv.weight",)
    assert checkpoint.state_dict["encoder.layers.0.conv.up_conv.weight"].shape == (4, 1, 3)
    assert checkpoint.config.model_type == "granite_speech"


def test_granite_checkpoint_alignment_report_splits_missing_and_shape_mismatch():
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
