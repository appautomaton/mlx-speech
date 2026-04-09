import pytest
from mlx_speech.models.fish_s2_pro import FishS2ProConfig


def test_config_defaults():
    cfg = FishS2ProConfig()
    assert cfg.vocab_size == 4096
    assert cfg.num_codebooks == 10
    assert cfg.slow_ar_dim == 2048
    assert cfg.fast_ar_dim == 1024


def test_config_from_huggingface():
    cfg = FishS2ProConfig.from_huggingface("fishaudio/s2-pro")
    assert cfg.model_dir == "fishaudio/s2-pro"
