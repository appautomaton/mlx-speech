import pytest
from mlx_speech.generation.fish_s2_pro import (
    FishS2ProOutput,
    generate_fish_s2_pro,
)


def test_output_dataclass():
    """Test output dataclass."""
    output = FishS2ProOutput(
        waveform=None,
        sample_rate=22050,
        generated_tokens=0,
    )
    assert output.sample_rate == 22050


def test_generate_returns_output():
    """Test generate returns correct type."""
    pytest.skip("Requires model checkpoint")
