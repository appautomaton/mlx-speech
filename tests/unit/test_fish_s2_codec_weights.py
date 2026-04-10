import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fish_s2_pro.codec_config import FishCodecConfig
from mlx_speech.models.fish_s2_pro.codec_weights import save_codec_assets


def test_codec_config_defaults_match_reference():
    cfg = FishCodecConfig()
    assert cfg.to_dict() == {
        "sample_rate": 44100,
        "latent_dim": 1024,
        "semantic_codebook_size": 4096,
        "n_codebooks": 9,
        "decoder_dim": 1536,
        "encoder_dim": 64,
    }


def test_codec_config_is_immutable():
    cfg = FishCodecConfig()

    with pytest.raises(FrozenInstanceError):
        cfg.decoder_dim = 2048


def test_save_codec_assets_writes_config_and_weights(tmp_path):
    weights = {
        "decoder.test.weight": mx.ones((2, 2), dtype=mx.float32),
        "decoder.bias": mx.zeros((2,), dtype=mx.float16),
    }
    cfg = FishCodecConfig()

    save_codec_assets(tmp_path, weights, cfg)

    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "model.safetensors"

    assert config_path.is_file()
    assert weights_path.is_file()
    assert json.loads(config_path.read_text(encoding="utf-8")) == cfg.to_dict()

    restored = mx.load(str(weights_path))
    assert set(restored) == set(weights)
    for name, tensor in weights.items():
        assert restored[name].shape == tensor.shape
        assert restored[name].dtype == tensor.dtype
        assert restored[name].tolist() == tensor.tolist()


def test_save_codec_assets_keeps_previous_assets_if_weight_write_fails(
    tmp_path, monkeypatch
):
    original_weights = {"decoder.test.weight": mx.ones((1, 1), dtype=mx.float32)}
    original_cfg = FishCodecConfig()
    save_codec_assets(tmp_path, original_weights, original_cfg)

    replacement_weights = {"decoder.test.weight": mx.zeros((3, 1), dtype=mx.float16)}
    replacement_cfg = FishCodecConfig(decoder_dim=2048)
    original_save = mx.save_safetensors

    def _failing_save(path: str, weights: dict[str, mx.array]) -> None:
        original_save(path, weights)
        raise RuntimeError("boom")

    monkeypatch.setattr(mx, "save_safetensors", _failing_save)

    with pytest.raises(RuntimeError, match="boom"):
        save_codec_assets(tmp_path, replacement_weights, replacement_cfg)

    assert (
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        == original_cfg.to_dict()
    )
    restored = mx.load(str(tmp_path / "model.safetensors"))
    assert restored["decoder.test.weight"].shape == (1, 1)
    assert restored["decoder.test.weight"].dtype == mx.float32
    assert restored["decoder.test.weight"].tolist() == [[1.0]]


def test_save_codec_assets_keeps_old_config_if_config_replace_fails(
    tmp_path, monkeypatch
):
    original_weights = {"decoder.test.weight": mx.array([[1.0, 2.0]], dtype=mx.float32)}
    original_cfg = FishCodecConfig()
    save_codec_assets(tmp_path, original_weights, original_cfg)

    replacement_weights = {
        "decoder.test.weight": mx.array([[5.0, 8.0]], dtype=mx.float32)
    }
    replacement_cfg = FishCodecConfig(decoder_dim=2048)
    original_replace = Path.replace

    def _failing_replace(self: Path, target: Path) -> Path:
        if self.name == ".config.json.tmp":
            raise RuntimeError("config replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _failing_replace)

    with pytest.raises(RuntimeError, match="config replace failed"):
        save_codec_assets(tmp_path, replacement_weights, replacement_cfg)

    assert (
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        == original_cfg.to_dict()
    )
    restored = mx.load(str(tmp_path / "model.safetensors"))
    assert restored["decoder.test.weight"].tolist() == [[5.0, 8.0]]


def test_save_codec_assets_rejects_empty_weight_dict(tmp_path):
    with pytest.raises(ValueError, match="No codec weights"):
        save_codec_assets(tmp_path, {}, FishCodecConfig())
