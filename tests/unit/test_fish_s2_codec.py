from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.fish_s2_pro.codec import FishS2Codec, MissingCodecAssetError
from mlx_speech.models.fish_s2_pro.codec_config import FishCodecConfig
from mlx_speech.models.fish_s2_pro.codec_model import FishCodecModel, build_ae
from mlx_speech.models.fish_s2_pro.codec_weights import save_codec_assets


def _write_real_codec_assets(
    tmp_path, config: FishCodecConfig | None = None
) -> FishCodecConfig:
    config = config or FishCodecConfig()
    model = build_ae(**config.to_dict())
    weights = tree_flatten(model.parameters(), destination={})
    save_codec_assets(tmp_path, weights, config)
    return config


def _to_upstream_weight_norm_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    upstream = {}
    for key, value in weights.items():
        if key.endswith(".weight_g"):
            key = key[: -len(".weight_g")] + ".conv.parametrizations.weight.original0"
        elif key.endswith(".weight_v"):
            key = key[: -len(".weight_v")] + ".conv.parametrizations.weight.original1"
        elif key.endswith(".bias"):
            prefix = key[: -len(".bias")]
            if prefix + ".weight_g" in weights:
                key = prefix + ".conv.bias"
        upstream[key] = value
    return upstream


def _drop_causal_conv_alias_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    stripped = {}
    for key, value in weights.items():
        if key.endswith(".conv.conv.weight") or key.endswith(".conv.conv.bias"):
            continue
        stripped[key] = value
    return stripped


def test_codec_requires_local_asset_dir(tmp_path):
    with pytest.raises(MissingCodecAssetError, match="config.json, model.safetensors"):
        FishS2Codec.from_dir(tmp_path / "missing")


def test_codec_requires_weights_when_config_exists(tmp_path):
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text("{}")

    with pytest.raises(MissingCodecAssetError, match="model.safetensors"):
        FishS2Codec.from_dir(codec_dir)


def test_codec_rejects_directory_instead_of_required_file(tmp_path):
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    (codec_dir / "config.json").mkdir()
    (codec_dir / "model.safetensors").write_text("")

    with pytest.raises(MissingCodecAssetError, match="config.json"):
        FishS2Codec.from_dir(codec_dir)


def test_codec_model_uses_same_asset_contract_message(tmp_path):
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text("{}")
    (codec_dir / "model.safetensors").mkdir()

    with pytest.raises(MissingCodecAssetError, match="model.safetensors"):
        FishCodecModel.from_dir(codec_dir)


def test_codec_decode_uses_model_boundary(monkeypatch, tmp_path):
    seen = {}

    class _FakeCodecModel:
        def decode(self, codes):
            seen["codes"] = codes
            return mx.ones((1, 44100), dtype=mx.float32)

    monkeypatch.setattr(
        FishS2Codec, "_load_model", lambda self, path: _FakeCodecModel()
    )
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text("{}")
    (codec_dir / "model.safetensors").write_text("")
    codec = FishS2Codec.from_dir(codec_dir)
    codes = mx.zeros((10, 4), dtype=mx.int32)
    audio = codec.decode(codes)
    assert seen["codes"] is codes
    assert audio.shape == (1, 44100)


def test_codec_model_loads_converted_assets(tmp_path):
    _write_real_codec_assets(tmp_path)
    codec = FishS2Codec.from_dir(tmp_path)
    assert codec.sample_rate == 44100


def test_codec_model_loads_sample_rate_from_assets(tmp_path):
    _write_real_codec_assets(tmp_path, FishCodecConfig(sample_rate=48000))
    codec = FishS2Codec.from_dir(tmp_path)
    assert codec.sample_rate == 48000


def test_codec_model_rejects_partial_weights(tmp_path):
    save_codec_assets(
        tmp_path,
        {"decoder.test.weight": mx.ones((2, 2), dtype=mx.float32)},
        FishCodecConfig(),
    )

    with pytest.raises(ValueError, match="codec weights do not match model parameters"):
        FishS2Codec.from_dir(tmp_path)


def test_codec_decode_rejects_wrong_codebook_count(tmp_path):
    _write_real_codec_assets(tmp_path)
    codec = FishS2Codec.from_dir(tmp_path)
    with pytest.raises(ValueError, match="expected 10 Fish codebook rows"):
        codec.decode(mx.zeros((9, 4), dtype=mx.int32))


def test_codec_decode_derives_expected_codebook_rows_from_config(tmp_path):
    config = FishCodecConfig(n_codebooks=2)
    _write_real_codec_assets(tmp_path, config)
    codec = FishS2Codec.from_dir(tmp_path)

    with pytest.raises(ValueError, match="expected 3 Fish codebook rows"):
        codec.decode(mx.zeros((2, 4), dtype=mx.int32))


def test_codec_model_loads_upstream_style_weight_norm_keys(tmp_path):
    config = FishCodecConfig()
    model = build_ae(**config.to_dict())
    weights = tree_flatten(model.parameters(), destination={})
    upstream_weights = _to_upstream_weight_norm_keys(weights)
    save_codec_assets(tmp_path, upstream_weights, config)

    codec = FishS2Codec.from_dir(tmp_path)

    assert codec.sample_rate == config.sample_rate


def test_codec_model_loads_upstream_weights_without_causal_conv_aliases(tmp_path):
    config = FishCodecConfig()
    model = build_ae(**config.to_dict())
    weights = tree_flatten(model.parameters(), destination={})
    upstream_weights = _drop_causal_conv_alias_keys(
        _to_upstream_weight_norm_keys(weights)
    )
    save_codec_assets(tmp_path, upstream_weights, config)

    codec = FishS2Codec.from_dir(tmp_path)

    assert codec.sample_rate == config.sample_rate


def test_codec_decode_rejects_out_of_range_codes(tmp_path):
    config = FishCodecConfig(n_codebooks=2)
    _write_real_codec_assets(tmp_path, config)
    codec = FishS2Codec.from_dir(tmp_path)
    codes = mx.zeros((3, 4), dtype=mx.int32)
    codes[0, 0] = config.semantic_codebook_size

    with pytest.raises(ValueError, match="semantic code IDs out of range"):
        codec.decode(codes)


def test_codec_model_decode_runs_real_runtime_path(tmp_path):
    _write_real_codec_assets(tmp_path)

    codec = FishS2Codec.from_dir(tmp_path)
    audio = codec.decode(mx.zeros((10, 2), dtype=mx.int32))

    assert audio.ndim == 2
    assert audio.shape[0] == 1
    assert audio.shape[1] > 0
    assert mx.all(mx.isfinite(audio)).item()


def test_codec_model_decode_trims_to_reported_audio_length():
    class _FakeInnerModel:
        quantizer = type(
            "Q",
            (),
            {
                "semantic_quantizer": type("SQ", (), {"codebook_size": 4})(),
                "quantizer": type("RQ", (), {"codebook_size": 8})(),
            },
        )()

        def decode(self, batch_codes, feature_lengths):
            assert feature_lengths.tolist() == [4]
            return mx.ones((1, 1, 12), dtype=mx.float32), mx.array([5], dtype=mx.int32)

    model = FishCodecModel(
        model=_FakeInnerModel(),
        config=FishCodecConfig(semantic_codebook_size=4, n_codebooks=9),
        sample_rate=44100,
    )

    audio = model.decode(mx.zeros((10, 4), dtype=mx.int32))

    assert audio.shape == (1, 5)


def test_codec_model_decode_delegates_for_valid_codes():
    seen = {}

    class _FakeInnerModel:
        quantizer = type(
            "Q",
            (),
            {
                "semantic_quantizer": type("SQ", (), {"codebook_size": 4096})(),
                "quantizer": type("RQ", (), {"codebook_size": 1024})(),
            },
        )()

        def decode(self, batch_codes, feature_lengths):
            seen["codes"] = batch_codes
            seen["lengths"] = feature_lengths
            return mx.ones((1, 1, 8), dtype=mx.float32), mx.array([8], dtype=mx.int32)

    model = FishCodecModel(
        model=_FakeInnerModel(),
        config=FishCodecConfig(),
        sample_rate=44100,
    )

    audio = model.decode(mx.zeros((10, 4), dtype=mx.int32))

    assert seen["codes"].shape == (1, 10, 4)
    assert seen["lengths"].tolist() == [4]
    assert audio.shape == (1, 8)


def test_codec_model_decode_clamps_out_of_range_residual_codes():
    seen = {}

    class _FakeInnerModel:
        quantizer = type(
            "Q",
            (),
            {
                "semantic_quantizer": type("SQ", (), {"codebook_size": 4096})(),
                "quantizer": type("RQ", (), {"codebook_size": 1024})(),
            },
        )()

        def decode(self, batch_codes, feature_lengths):
            seen["codes"] = batch_codes
            seen["lengths"] = feature_lengths
            return mx.ones((1, 1, 8), dtype=mx.float32), mx.array([8], dtype=mx.int32)

    model = FishCodecModel(
        model=_FakeInnerModel(),
        config=FishCodecConfig(),
        sample_rate=44100,
    )

    codes = mx.array(
        [
            [12, 34, 56, 78],
            [0, 1023, 1200, 4095],
            [5, 2048, 999, 5000],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=mx.int32,
    )

    audio = model.decode(codes)

    assert seen["lengths"].tolist() == [4]
    assert seen["codes"][0, 1].tolist() == [0, 1023, 1023, 1023]
    assert seen["codes"][0, 2].tolist() == [5, 1023, 999, 1023]
    assert audio.shape == (1, 8)
