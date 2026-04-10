"""LongCat AudioDiT checkpoint tests using local downloaded assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.longcat_audiodit.checkpoint import load_longcat_checkpoint
from mlx_speech.models.longcat_audiodit.config import LongCatAudioDiTConfig

pytestmark = pytest.mark.checkpoint

LONGCAT_ORIGINAL_DIR = Path("models/longcat_audiodit/original")
skip_no_longcat = pytest.mark.skipif(
    not (LONGCAT_ORIGINAL_DIR / "model.safetensors").exists(),
    reason="No local LongCat original checkpoint available",
)


@skip_no_longcat
def test_longcat_config_parses_downloaded_config() -> None:
    config = LongCatAudioDiTConfig.from_path(LONGCAT_ORIGINAL_DIR)

    assert config.model_type == "audiodit"
    assert config.sampling_rate == 24000
    assert config.latent_hop == 2048
    assert config.dit_dim == 2560
    assert config.text_encoder_model == "google/umt5-base"
    assert config.text_encoder_config.d_model == 768
    assert config.vae_config.latent_dim == 64
    assert config.vae_config.scale == pytest.approx(0.71)


@skip_no_longcat
def test_longcat_checkpoint_loads_downloaded_weights() -> None:
    checkpoint = load_longcat_checkpoint(LONGCAT_ORIGINAL_DIR)

    assert checkpoint.key_count > 1200
    assert checkpoint.model_dir == LONGCAT_ORIGINAL_DIR
    assert checkpoint.state_dict["text_encoder.encoder.embed_tokens.weight"].shape == (
        256384,
        768,
    )
    assert checkpoint.state_dict[
        "transformer.blocks.0.self_attn.to_q.weight"
    ].shape == (2560, 2560)
    assert checkpoint.state_dict[
        "transformer.text_conv_layer.0.dwconv.weight"
    ].shape == (2560, 7, 1)
    assert checkpoint.state_dict["vae.encoder.layers.0.weight"].shape == (128, 7, 1)
