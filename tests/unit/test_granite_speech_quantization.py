from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_speech.models.granite_speech_asr import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechModel,
    GraniteSpeechProjectorConfig,
    GraniteSpeechTextConfig,
    QuantizationConfig,
    get_quantization_config,
    load_checkpoint_into_model,
    load_granite_speech_checkpoint,
    quantize_granite_speech_model,
    save_granite_speech_model,
)
from scripts.convert.granite_speech_asr import parse_args


def _quantizable_config() -> GraniteSpeechConfig:
    return GraniteSpeechConfig(
        encoder=GraniteSpeechEncoderConfig(
            input_dim=8,
            hidden_dim=64,
            output_dim=16,
            num_layers=2,
            num_heads=4,
            dim_head=16,
            feedforward_mult=2,
            conv_expansion_factor=2,
            conv_kernel_size=3,
            context_size=4,
            max_pos_emb=8,
            dropout=0.0,
        ),
        projector=GraniteSpeechProjectorConfig(
            hidden_size=64,
            encoder_hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            layer_norm_eps=1e-6,
        ),
        text=GraniteSpeechTextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        ),
        audio_token_index=127,
        window_size=4,
        downsample_rate=2,
    )


def test_granite_quantization_config_round_trip_and_community_alias() -> None:
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")
    payload = _quantizable_config().to_dict()
    payload["quantization_config"] = quantization.to_dict()

    config = GraniteSpeechConfig.from_dict(payload)

    assert QuantizationConfig.from_dict(quantization.to_dict()) == quantization
    assert get_quantization_config(config) == quantization


def test_granite_rejects_disagreeing_quantization_aliases() -> None:
    payload = _quantizable_config().to_dict()
    payload["quantization"] = QuantizationConfig().to_dict()
    payload["quantization_config"] = QuantizationConfig(group_size=32).to_dict()

    config = GraniteSpeechConfig.from_dict(payload)

    with pytest.raises(ValueError, match="blocks disagree"):
        get_quantization_config(config)


def test_granite_converter_defaults_match_release_layout() -> None:
    args = parse_args([])

    assert args.input_dir.as_posix().endswith(
        "models/ibm/granite_4_0_1b_speech/original"
    )
    assert args.output_dir.as_posix().endswith(
        "models/ibm/granite_4_0_1b_speech/mlx-int8"
    )
    assert (args.bits, args.group_size, args.mode) == (8, 64, "affine")


def test_granite_selective_quantization_keeps_acoustic_modules_bf16() -> None:
    model = GraniteSpeechModel(_quantizable_config())
    quantization = QuantizationConfig()

    quantize_granite_speech_model(model, quantization)

    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(
        model.language_model.model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(model.language_model.lm_head, nn.QuantizedLinear)
    assert isinstance(model.encoder.input_linear, nn.Linear)
    assert isinstance(model.projector.linear, nn.Linear)


def test_granite_affine_int8_save_reload_and_forward(tmp_path) -> None:
    config = _quantizable_config()
    quantization = QuantizationConfig()
    model = GraniteSpeechModel(config)
    model.set_dtype(mx.bfloat16)
    quantize_granite_speech_model(model, quantization)
    save_granite_speech_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )

    checkpoint = load_granite_speech_checkpoint(tmp_path)
    reloaded = GraniteSpeechModel(checkpoint.config)
    restored = get_quantization_config(checkpoint.config)
    assert restored == quantization
    quantize_granite_speech_model(
        reloaded,
        restored,
        state_dict=checkpoint.state_dict,
    )
    report = load_checkpoint_into_model(reloaded, checkpoint, strict=True)
    reloaded.set_dtype(mx.bfloat16)

    logits = reloaded(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(logits)

    assert report.is_exact_match
    assert logits.shape == (1, 3, 128)
    assert mx.all(mx.isfinite(logits)).item()


def test_granite_saved_artifact_copies_only_runtime_assets(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "README.md").write_text("upstream card", encoding="utf-8")
    (source / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    mx.save_safetensors(
        str(source / "model-00001-of-00001.safetensors"),
        {"ignored": mx.zeros((1,))},
    )

    config = _quantizable_config()
    quantization = QuantizationConfig()
    model = GraniteSpeechModel(config)
    quantize_granite_speech_model(model, quantization)
    save_granite_speech_model(
        model,
        output,
        config=config,
        quantization=quantization,
        copy_supporting_files_from=source,
    )

    payload = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert payload["quantization"] == quantization.to_dict()
    assert (output / "tokenizer.json").is_file()
    assert not (output / "README.md").exists()
    assert not (output / "model.safetensors.index.json").exists()
    assert not (output / "model-00001-of-00001.safetensors").exists()
