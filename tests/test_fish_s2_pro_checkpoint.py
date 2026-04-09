import pytest
from pathlib import Path
from mlx_speech.models.fish_s2_pro.checkpoint import (
    load_fish_s2_pro_checkpoint,
    FishS2ProCheckpoint,
)


def test_checkpoint_structure():
    """Test checkpoint has expected structure."""
    checkpoint_dir = Path("models/fish_s2_pro/mlx-int8")
    if not checkpoint_dir.exists():
        pytest.skip("No checkpoint available")

    ckpt = load_fish_s2_pro_checkpoint(str(checkpoint_dir))
    assert isinstance(ckpt, FishS2ProCheckpoint)
    assert hasattr(ckpt, "state_dict")
    assert hasattr(ckpt, "config")


def test_checkpoint_keys():
    """Test checkpoint contains expected keys."""
    checkpoint_dir = Path("models/fish_s2_pro/mlx-int8")
    if not checkpoint_dir.exists():
        pytest.skip("No checkpoint available")

    ckpt = load_fish_s2_pro_checkpoint(str(checkpoint_dir))
    keys = list(ckpt.state_dict.keys())
    assert len(keys) > 0
