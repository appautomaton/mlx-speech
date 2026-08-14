"""VibeVoice configuration coverage against local upstream assets."""

from pathlib import Path

import pytest

from mlx_speech.models.vibevoice.config import VibeVoiceConfig


MODEL_DIR = Path("models/vibevoice/original")

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(
        not (MODEL_DIR / "config.json").is_file(),
        reason="VibeVoice original config is not present",
    ),
]


def test_from_path() -> None:
    config = VibeVoiceConfig.from_path(MODEL_DIR)

    assert config.model_type == "vibevoice"
    assert config.hidden_size == 3584
    assert config.language_config.num_hidden_layers == 28
    assert config.acoustic_tokenizer_config.vae_dim == 64
    assert config.semantic_tokenizer_config.vae_dim == 128
    assert config.diffusion_config.head_layers == 4


def test_round_trip() -> None:
    config = VibeVoiceConfig.from_path(MODEL_DIR)
    restored = VibeVoiceConfig.from_dict(config.to_dict())

    assert restored.hidden_size == config.hidden_size
    assert (
        restored.acoustic_tokenizer_config.vae_dim
        == config.acoustic_tokenizer_config.vae_dim
    )
