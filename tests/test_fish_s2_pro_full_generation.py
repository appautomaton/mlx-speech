import pytest
from pathlib import Path
from mlx_speech.generation.fish_s2_pro import (
    generate_fish_s2_pro,
    FishS2ProModel,
)


def test_generate_returns_waveform():
    """Generation should produce waveform or indicate needs model."""
    if not Path("models/fish_s2_pro/mlx-int8").exists():
        pytest.skip("No model checkpoint")

    result = generate_fish_s2_pro("Hello world")
    assert result is not None
    assert result.sample_rate == 22050


def test_model_class_exists():
    """FishS2ProModel should exist."""
    from mlx_speech.generation.fish_s2_pro import FishS2ProModel

    assert FishS2ProModel is not None


def test_fish_s2_pro_output_dataclass():
    """FishS2ProOutput should be a dataclass with expected fields."""
    from mlx_speech.generation.fish_s2_pro import FishS2ProOutput

    result = FishS2ProOutput(waveform=None, sample_rate=22050, generated_tokens=0)
    assert result.waveform is None
    assert result.sample_rate == 22050
    assert result.generated_tokens == 0
