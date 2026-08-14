"""Local VibeVoice checkpoint loading and alignment coverage."""

from pathlib import Path

import pytest

from mlx_speech.models.vibevoice.checkpoint import (
    load_vibevoice_checkpoint,
    load_vibevoice_model,
)


MODEL_DIR = Path("models/vibevoice/mlx-int8")
HAS_CHECKPOINT = MODEL_DIR.is_dir() and any(MODEL_DIR.glob("*.safetensors"))

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(
        not HAS_CHECKPOINT,
        reason="VibeVoice checkpoint is not present",
    ),
]


def test_load_checkpoint() -> None:
    checkpoint = load_vibevoice_checkpoint(MODEL_DIR)

    assert checkpoint.key_count > 0
    assert checkpoint.config.model_type == "vibevoice"


def test_model_alignment() -> None:
    loaded = load_vibevoice_model(MODEL_DIR, strict=False)

    assert loaded.alignment_report.is_exact_match
