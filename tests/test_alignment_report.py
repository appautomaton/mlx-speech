import json
from pathlib import Path

import mlx.core as mx

from mlx_voice.models.moss_local import (
    MossTTSLocalModel,
    MossTTSLocalConfig,
    load_checkpoint_into_model,
    load_moss_tts_local_checkpoint,
    validate_checkpoint_against_model,
)


def _tiny_checkpoint_dir(tmp_path: Path) -> Path:
    config = {
        "n_vq": 2,
        "audio_vocab_size": 16,
        "local_hidden_size": 12,
        "local_num_layers": 2,
        "language_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "max_position_embeddings": 64,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = MossTTSLocalModel(MossTTSLocalConfig.from_dict(config))
    model.save_weights(str(tmp_path / "weights.safetensors"))
    return tmp_path


def test_alignment_report_exact_match_for_saved_weights(tmp_path: Path) -> None:
    model_dir = _tiny_checkpoint_dir(tmp_path)
    config = MossTTSLocalConfig.from_path(model_dir)
    model = MossTTSLocalModel(config)
    checkpoint = load_moss_tts_local_checkpoint(model_dir)

    report = validate_checkpoint_against_model(model, checkpoint)

    assert report.is_exact_match
    assert report.missing_in_model == ()
    assert report.missing_in_checkpoint == ()
    assert report.shape_mismatches == ()


def test_load_checkpoint_into_model_strict_roundtrip(tmp_path: Path) -> None:
    model_dir = _tiny_checkpoint_dir(tmp_path)
    config = MossTTSLocalConfig.from_path(model_dir)
    model = MossTTSLocalModel(config)
    checkpoint = load_moss_tts_local_checkpoint(model_dir)

    report = load_checkpoint_into_model(model, checkpoint, strict=True)

    assert report.is_exact_match


def test_alignment_report_catches_shape_mismatch(tmp_path: Path) -> None:
    model_dir = _tiny_checkpoint_dir(tmp_path)
    config = MossTTSLocalConfig.from_path(model_dir)
    model = MossTTSLocalModel(config)
    checkpoint = load_moss_tts_local_checkpoint(model_dir)

    bad_state = dict(checkpoint.state_dict)
    bad_state["model.embedding_list.0.weight"] = mx.ones((127, 32))
    bad_checkpoint = checkpoint.__class__(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        state_dict=bad_state,
        source_files=checkpoint.source_files,
        skipped_keys=checkpoint.skipped_keys,
        renamed_keys=checkpoint.renamed_keys,
    )

    report = validate_checkpoint_against_model(model, bad_checkpoint)

    assert not report.is_exact_match
    assert report.shape_mismatches
