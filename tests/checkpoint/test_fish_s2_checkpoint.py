from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fish_s2_pro.checkpoint import (
    load_fish_s2_pro_checkpoint,
    validate_checkpoint_against_model,
)
from mlx_speech.models.fish_s2_pro.codec_weights import convert_codec_pth_to_assets
from mlx_speech.models.fish_s2_pro.model import DualARTransformer


def test_load_checkpoint_reads_original_fish_s2_shards():
    model_dir = Path("models/fish_s2_pro/original")
    if not model_dir.exists():
        pytest.skip("No local Fish S2 checkpoint")
    ckpt = load_fish_s2_pro_checkpoint(model_dir)

    assert "text_model.model.embeddings.weight" not in ckpt.state_dict
    assert "embeddings.weight" in ckpt.state_dict
    assert "audio_decoder.codebook_embeddings.weight" not in ckpt.state_dict
    assert "codebook_embeddings.weight" in ckpt.state_dict
    assert "audio_decoder.output.weight" not in ckpt.state_dict
    assert "fast_output.weight" in ckpt.state_dict

    fast_layer_keys = [key for key in ckpt.state_dict if key.startswith("fast_layers.")]
    assert fast_layer_keys
    assert not any(key.startswith("audio_decoder.layers.") for key in ckpt.state_dict)
    assert not any(key.startswith("audio_decoder.") for key in ckpt.state_dict)

    model = DualARTransformer(ckpt.config)
    report = validate_checkpoint_against_model(model, ckpt)
    assert report.is_exact_match


def test_convert_local_codec_archive_without_torch(tmp_path: Path):
    codec_pth = Path("models/fish_s2_pro/original/codec.pth")
    if not codec_pth.is_file():
        pytest.skip("No local Fish codec archive")

    convert_codec_pth_to_assets(codec_pth, tmp_path)

    weights_path = tmp_path / "model.safetensors"
    assert (tmp_path / "config.json").is_file()
    assert weights_path.is_file()

    weights = mx.load(str(weights_path))
    assert "encoder.block.0.conv.bias" in weights
