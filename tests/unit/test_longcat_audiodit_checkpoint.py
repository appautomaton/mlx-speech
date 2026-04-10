from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_speech.models.longcat_audiodit.checkpoint import (
    AlignmentReport,
    LongCatCheckpoint,
    LoadedLongCatModel,
    load_checkpoint_into_model,
    load_longcat_checkpoint,
    load_longcat_model,
    save_longcat_model,
    resolve_longcat_model_dir,
    resolve_longcat_tokenizer_dir,
    sanitize_state_dict,
    validate_checkpoint_against_model,
)
from mlx_speech.models.longcat_audiodit.config import (
    LongCatAudioDiTConfig,
    QuantizationConfig,
)


def _write_config(tmp_path: Path, *, quantized: bool = False) -> None:
    payload = {
        "model_type": "audiodit",
        "sampling_rate": 24000,
        "latent_hop": 2048,
        "latent_dim": 64,
        "max_wav_duration": 60,
        "dit_dim": 2560,
        "dit_depth": 32,
        "dit_heads": 32,
        "dit_ff_mult": 3.6,
        "text_encoder_model": "google/umt5-base",
        "text_encoder_config": {
            "d_model": 768,
            "d_ff": 2048,
            "d_kv": 64,
            "num_heads": 12,
            "num_layers": 12,
            "num_decoder_layers": 12,
            "vocab_size": 256384,
        },
        "vae_config": {
            "in_channels": 1,
            "channels": 128,
            "c_mults": [1, 2],
            "strides": [2, 4],
            "latent_dim": 64,
            "encoder_latent_dim": 128,
            "scale": 0.71,
        },
    }
    if quantized:
        payload["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_model_dir_prefers_int8_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.MODELS_ROOT", tmp_path
    )
    base = tmp_path / "longcat_audiodit"
    (base / "original").mkdir(parents=True)
    (base / "mlx-int8").mkdir(parents=True)
    (base / "mlx-int8" / "model.safetensors").write_bytes(b"x")

    assert resolve_longcat_model_dir(prefer_mlx_int8=True) == base / "mlx-int8"
    assert resolve_longcat_model_dir(prefer_mlx_int8=False) == base / "original"


def test_resolve_tokenizer_dir_uses_local_umt5_layout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.MODELS_ROOT", tmp_path
    )
    assert (
        resolve_longcat_tokenizer_dir()
        == tmp_path / "longcat_audiodit" / "tokenizer" / "umt5-base"
    )


def test_sanitize_state_dict_folds_weight_norm_conv1d_and_convtranspose() -> None:
    weights = {
        "vae.encoder.layers.0.weight_g": mx.array([[[2.0]], [[3.0]]], dtype=mx.float32),
        "vae.encoder.layers.0.weight_v": mx.ones((2, 3, 4), dtype=mx.float32),
        "vae.decoder.layers.1.layers.1.weight_g": mx.array(
            [[[5.0]], [[7.0]], [[11.0]], [[13.0]]], dtype=mx.float32
        ),
        "vae.decoder.layers.1.layers.1.weight_v": mx.ones((4, 2, 5), dtype=mx.float32),
        "vae.encoder.layers.0.bias": mx.zeros((2,), dtype=mx.float32),
        "transformer.rotary_emb.inv_freq": mx.zeros((16,), dtype=mx.float32),
    }

    sanitized, skipped, renamed = sanitize_state_dict(weights, is_mlx_native=False)

    assert sanitized["vae.encoder.layers.0.weight"].shape == (2, 4, 3)
    assert sanitized["vae.decoder.layers.1.layers.1.weight"].shape == (2, 5, 4)
    assert sanitized["vae.encoder.layers.0.bias"].shape == (2,)
    assert "transformer.rotary_emb.inv_freq" not in sanitized
    assert skipped == ("transformer.rotary_emb.inv_freq",)
    assert ("vae.encoder.layers.0.weight_g", "vae.encoder.layers.0.weight") in renamed
    assert (
        "vae.decoder.layers.1.layers.1.weight_g",
        "vae.decoder.layers.1.layers.1.weight",
    ) in renamed


def test_sanitize_state_dict_only_treats_top_level_decoder_conv_as_transpose() -> None:
    weights = {
        "vae.decoder.layers.1.layers.1.weight_g": mx.array(
            [[[2.0]], [[3.0]], [[5.0]], [[7.0]]],
            dtype=mx.float32,
        ),
        "vae.decoder.layers.1.layers.1.weight_v": mx.ones((4, 2, 5), dtype=mx.float32),
        "vae.decoder.layers.1.layers.2.layers.1.weight_g": mx.array(
            [[[11.0]], [[13.0]], [[17.0]]],
            dtype=mx.float32,
        ),
        "vae.decoder.layers.1.layers.2.layers.1.weight_v": mx.ones(
            (3, 2, 7), dtype=mx.float32
        ),
    }

    sanitized, _, _ = sanitize_state_dict(weights, is_mlx_native=False)

    assert sanitized["vae.decoder.layers.1.layers.1.weight"].shape == (2, 5, 4)
    assert sanitized["vae.decoder.layers.1.layers.2.layers.1.weight"].shape == (3, 7, 2)


def test_sanitize_state_dict_keeps_mlx_native_weights_as_is() -> None:
    weights = {
        "transformer.blocks.0.self_attn.to_q.weight": mx.zeros(
            (2560, 2560), dtype=mx.float32
        ),
        "vae.encoder.layers.0.weight": mx.zeros((128, 7, 1), dtype=mx.float32),
    }

    sanitized, skipped, renamed = sanitize_state_dict(weights, is_mlx_native=True)

    assert sanitized == weights
    assert skipped == ()
    assert renamed == ()


def test_load_longcat_checkpoint_uses_config_and_sanitized_weights(
    tmp_path: Path, monkeypatch
) -> None:
    _write_config(tmp_path, quantized=False)

    class _Loaded:
        def __init__(self) -> None:
            self.files = (tmp_path / "model.safetensors",)
            self.index = None
            self.weights = {
                "transformer.blocks.0.self_attn.to_q.weight": mx.zeros(
                    (2560, 2560), dtype=mx.float32
                ),
                "transformer.rotary_emb.inv_freq": mx.zeros((16,), dtype=mx.float32),
            }

    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.load_state_dict",
        lambda model_dir: _Loaded(),
    )

    checkpoint = load_longcat_checkpoint(tmp_path)

    assert isinstance(checkpoint, LongCatCheckpoint)
    assert checkpoint.model_dir == tmp_path
    assert checkpoint.key_count == 1
    assert checkpoint.skipped_keys == ("transformer.rotary_emb.inv_freq",)
    assert checkpoint.source_files == (tmp_path / "model.safetensors",)


class _FakeRuntimeModel:
    def __init__(self) -> None:
        self._params = {
            "layer.weight": mx.zeros((2, 2), dtype=mx.float32),
            "layer.scales": mx.zeros((2, 1), dtype=mx.float32),
        }
        self.loaded: tuple[list[tuple[str, mx.array]], bool] | None = None

    def parameters(self):
        return self._params

    def load_weights(self, file_or_weights, strict: bool = True):
        self.loaded = (list(file_or_weights), strict)


def test_validate_and_load_checkpoint_against_model() -> None:
    class _PlainFakeModel:
        def __init__(self) -> None:
            self._params = {"layer.weight": mx.zeros((2, 2), dtype=mx.float32)}
            self.loaded: tuple[list[tuple[str, mx.array]], bool] | None = None

        def parameters(self):
            return self._params

        def load_weights(self, file_or_weights, strict: bool = True):
            self.loaded = (list(file_or_weights), strict)

    model = _PlainFakeModel()
    checkpoint = LongCatCheckpoint(
        model_dir=Path("."),
        config=LongCatAudioDiTConfig(),
        state_dict={"layer.weight": mx.ones((2, 2), dtype=mx.float32)},
        source_files=(),
        skipped_keys=(),
        renamed_keys=(),
    )

    report = validate_checkpoint_against_model(model, checkpoint)
    assert report == AlignmentReport(
        missing_in_model=(), missing_in_checkpoint=(), shape_mismatches=()
    )

    loaded_report = load_checkpoint_into_model(model, checkpoint, strict=True)
    assert loaded_report.is_exact_match is True
    assert model.loaded is not None
    assert model.loaded[0][0][0] == "layer.weight"


def test_save_longcat_model_writes_weights_and_config(tmp_path: Path) -> None:
    model = _FakeRuntimeModel()
    output_dir = save_longcat_model(
        model,
        tmp_path / "mlx-int8",
        config=LongCatAudioDiTConfig(),
        quantization=QuantizationConfig(bits=8, group_size=64, mode="affine"),
    )

    assert output_dir == tmp_path / "mlx-int8"
    assert (output_dir / "model.safetensors").exists()
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert payload["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_load_longcat_model_builds_runtime_bundle(tmp_path: Path, monkeypatch) -> None:
    checkpoint = LongCatCheckpoint(
        model_dir=tmp_path,
        config=LongCatAudioDiTConfig(
            quantization=QuantizationConfig(bits=8, group_size=64, mode="affine")
        ),
        state_dict={
            "layer.weight": mx.ones((2, 2), dtype=mx.float32),
            "layer.scales": mx.ones((2, 1), dtype=mx.float32),
        },
        source_files=(tmp_path / "model.safetensors",),
        skipped_keys=(),
        renamed_keys=(),
    )
    fake_model = _FakeRuntimeModel()

    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.resolve_longcat_model_dir",
        lambda model_dir=None, prefer_mlx_int8=True: tmp_path,
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.load_longcat_checkpoint",
        lambda model_dir: checkpoint,
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.LongCatUMT5Encoder",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.LongCatAudioDiTTransformer",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.LongCatAudioDiTVae",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.LongCatAudioDiTModel",
        lambda config, text_encoder, transformer, vae: fake_model,
    )
    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.checkpoint.quantize_longcat_model",
        lambda model, quantization, state_dict=None: model,
    )

    loaded = load_longcat_model()

    assert isinstance(loaded, LoadedLongCatModel)
    assert loaded.model is fake_model
    assert loaded.quantization == checkpoint.config.quantization
    assert loaded.alignment_report.is_exact_match is True
