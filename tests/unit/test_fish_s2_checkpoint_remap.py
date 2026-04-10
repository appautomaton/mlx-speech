from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fish_s2_pro.checkpoint import (
    FishS2ProCheckpoint,
    FishS2ProConfig,
    load_fish_s2_pro_checkpoint,
    load_checkpoint_into_model,
    sanitize_state_dict,
    validate_checkpoint_against_model,
)


class _DummyModel:
    def __init__(self, params: dict[str, mx.array]):
        self._params = params
        self.loaded_weights = None

    def parameters(self):
        return self._params

    def load_weights(self, file_or_weights, strict: bool = True):
        self.loaded_weights = (list(file_or_weights), strict)


def _checkpoint(state_dict: dict[str, mx.array]) -> FishS2ProCheckpoint:
    return FishS2ProCheckpoint(
        model_dir=Path("models/fish_s2_pro/original"),
        state_dict=state_dict,
        config=FishS2ProConfig(model_dir="models/fish_s2_pro/original"),
        source_files=(),
        skipped_keys=(),
        renamed_keys=(),
    )


def test_sanitize_state_dict_remaps_upstream_keys():
    weights = {
        "text_model.model.embeddings.weight": mx.zeros((2, 2)),
        "audio_decoder.codebook_embeddings.weight": mx.zeros((2, 2)),
        "audio_decoder.layers.0.attention.wqkv.weight": mx.zeros((2, 2)),
        "audio_decoder.output.weight": mx.zeros((2, 2)),
        "audio_decoder.norm.weight": mx.zeros((2, 2)),
        "audio_decoder.attention.wq.weight": mx.zeros((2, 2)),
    }
    sanitized, skipped, renamed = sanitize_state_dict(weights)
    assert "embeddings.weight" in sanitized
    assert "codebook_embeddings.weight" in sanitized
    assert "fast_layers.0.attention.wqkv.weight" in sanitized
    assert "fast_output.weight" in sanitized
    assert "fast_norm.weight" in sanitized
    assert "fast_attention.wq.weight" in sanitized
    assert not skipped
    assert renamed


def test_sanitize_state_dict_rejects_duplicate_key_after_remap():
    weights = {
        "audio_decoder.output.weight": mx.zeros((2, 2)),
        "fast_output.weight": mx.ones((2, 2)),
    }

    with pytest.raises(
        ValueError, match="Duplicate key after sanitization: fast_output.weight"
    ):
        sanitize_state_dict(weights)


def test_validate_checkpoint_against_model_reports_missing_keys():
    model = _DummyModel({"embeddings.weight": mx.zeros((2, 2))})
    checkpoint = _checkpoint({"fast_output.weight": mx.zeros((2, 2))})

    report = validate_checkpoint_against_model(model, checkpoint)

    assert not report.is_exact_match
    assert report.missing_in_model == ("fast_output.weight",)
    assert report.missing_in_checkpoint == ("embeddings.weight",)
    assert report.shape_mismatches == ()


def test_validate_checkpoint_against_model_reports_shape_mismatch():
    model = _DummyModel({"embeddings.weight": mx.zeros((2, 2))})
    checkpoint = _checkpoint({"embeddings.weight": mx.zeros((3, 2))})

    report = validate_checkpoint_against_model(model, checkpoint)

    assert not report.is_exact_match
    assert report.missing_in_model == ()
    assert report.missing_in_checkpoint == ()
    assert report.shape_mismatches == (("embeddings.weight", (2, 2), (3, 2)),)


def test_load_checkpoint_into_model_rejects_partial_load_in_strict_mode():
    model = _DummyModel({"embeddings.weight": mx.zeros((2, 2))})
    checkpoint = _checkpoint({"fast_output.weight": mx.zeros((2, 2))})

    with pytest.raises(ValueError, match="Checkpoint/model alignment failed"):
        load_checkpoint_into_model(model, checkpoint, strict=True)

    assert model.loaded_weights is None


def test_load_checkpoint_into_model_allows_partial_load_in_non_strict_mode():
    model = _DummyModel({"embeddings.weight": mx.zeros((2, 2))})
    checkpoint = _checkpoint({"fast_output.weight": mx.zeros((2, 2))})

    report = load_checkpoint_into_model(model, checkpoint, strict=False)

    assert not report.is_exact_match
    assert model.loaded_weights == (
        [("fast_output.weight", checkpoint.state_dict["fast_output.weight"])],
        False,
    )


def test_load_checkpoint_requires_config_json(tmp_path: Path):
    weights = {"text_model.model.embeddings.weight": mx.zeros((2, 2))}
    mx.save_safetensors(tmp_path / "weights.safetensors", weights)

    with pytest.raises(FileNotFoundError, match="config.json"):
        load_fish_s2_pro_checkpoint(tmp_path)
