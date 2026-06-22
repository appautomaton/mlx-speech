from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.qwen3_asr.checkpoint import (
    QuantizationConfig,
    get_quantization_config,
    load_checkpoint_into_model,
    load_qwen3_asr_checkpoint,
    quantize_qwen3_asr_model,
    save_qwen3_asr_model,
)
from mlx_speech.models.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel


def _quantizable_config() -> Qwen3ASRConfig:
    """Tiny config whose Linear/Embedding dims divide by both 32 and 64,
    so affine (gs 64) and mxfp8 (gs 32) quantize the same layer set."""
    audio = Qwen3ASRAudioConfig(
        d_model=64,
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=8,
        encoder_ffn_dim=128,
        downsample_hidden_size=4,
        output_dim=64,
        max_source_positions=64,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    text = Qwen3ASRTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        eos_token_id=9,
        extra={"tie_word_embeddings": False},
    )
    return Qwen3ASRConfig(
        thinker_config=Qwen3ASRThinkerConfig(
            audio_config=audio,
            text_config=text,
            audio_token_id=31,
            audio_start_token_id=29,
            audio_end_token_id=30,
        ),
        support_languages=("Chinese", "English"),
    )


def _first_quantized_linear(model: nn.Module) -> nn.QuantizedLinear:
    for module in model.modules():
        if isinstance(module, nn.QuantizedLinear):
            return module
    raise AssertionError("no QuantizedLinear found in model")


def _param_keys(module: nn.Module) -> set[str]:
    return set(tree_flatten(module.parameters(), destination={}))


def _quantize_save_reload(tmp_path, *, mode: str, group_size: int):
    config = _quantizable_config()
    quantization = QuantizationConfig(bits=8, group_size=group_size, mode=mode)

    model = Qwen3ASRModel(config)
    quantize_qwen3_asr_model(model, quantization)
    save_qwen3_asr_model(model, tmp_path, config=config, quantization=quantization)

    checkpoint = load_qwen3_asr_checkpoint(tmp_path)
    reloaded = Qwen3ASRModel(checkpoint.config)
    restored_q = get_quantization_config(checkpoint.config)
    assert restored_q == quantization
    quantize_qwen3_asr_model(reloaded, restored_q, state_dict=checkpoint.state_dict)
    report = load_checkpoint_into_model(reloaded, checkpoint, strict=True)
    reloaded.set_dtype(mx.bfloat16)
    mx.eval(reloaded.parameters())
    return reloaded, report


# --------------------------------------------------------------------------- #
# Config / QuantizationConfig serialization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "quant",
    [
        QuantizationConfig(bits=8, group_size=64, mode="affine"),
        QuantizationConfig(bits=8, group_size=32, mode="mxfp8"),
    ],
)
def test_quantization_config_round_trip(quant):
    assert QuantizationConfig.from_dict(quant.to_dict()) == quant


def test_qwen3_asr_config_to_dict_round_trips_with_quantization_block():
    config = _quantizable_config()
    payload = config.to_dict()
    payload["quantization"] = QuantizationConfig(
        bits=8, group_size=32, mode="mxfp8"
    ).to_dict()

    restored = Qwen3ASRConfig.from_dict(payload)

    assert restored.model_type == "qwen3_asr"
    assert restored.text_config.hidden_size == 64
    assert restored.audio_config.output_dim == 64
    assert restored.support_languages == ("Chinese", "English")
    assert restored.text_config.extra.get("tie_word_embeddings") is False
    assert get_quantization_config(restored) == QuantizationConfig(
        bits=8, group_size=32, mode="mxfp8"
    )


def test_get_quantization_config_returns_none_without_block():
    assert get_quantization_config(_quantizable_config()) is None


# --------------------------------------------------------------------------- #
# quantize -> save -> reload round trips
# --------------------------------------------------------------------------- #


def test_affine_int8_round_trip(tmp_path):
    model, report = _quantize_save_reload(tmp_path, mode="affine", group_size=64)

    assert report.is_loadable_match
    assert any(isinstance(m, nn.QuantizedLinear) for m in model.modules())
    assert any(isinstance(m, nn.QuantizedEmbedding) for m in model.modules())
    # affine quantization carries a bias term
    assert "biases" in _param_keys(_first_quantized_linear(model))

    out = model.prefill(
        inputs_embeds=mx.zeros((1, 4, 64), dtype=mx.bfloat16), max_cache_len=8
    )
    mx.eval(out.logits)
    assert out.logits.shape == (1, 4, 128)
    assert mx.all(mx.isfinite(out.logits)).item()


def test_mxfp8_round_trip_has_no_bias(tmp_path):
    model, report = _quantize_save_reload(tmp_path, mode="mxfp8", group_size=32)

    assert report.is_loadable_match
    assert any(isinstance(m, nn.QuantizedLinear) for m in model.modules())
    # mxfp8 (microscaling FP8) has a shared scale per group but no bias term
    keys = _param_keys(_first_quantized_linear(model))
    assert "scales" in keys
    assert "biases" not in keys

    out = model.prefill(
        inputs_embeds=mx.zeros((1, 4, 64), dtype=mx.bfloat16), max_cache_len=8
    )
    mx.eval(out.logits)
    assert mx.all(mx.isfinite(out.logits)).item()


def test_saved_config_json_carries_quantization_block(tmp_path):
    config = _quantizable_config()
    quantization = QuantizationConfig(bits=8, group_size=32, mode="mxfp8")
    model = Qwen3ASRModel(config)
    quantize_qwen3_asr_model(model, quantization)
    save_qwen3_asr_model(model, tmp_path, config=config, quantization=quantization)

    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert payload["quantization"] == {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    # config.json remains a valid Qwen3-ASR config
    assert Qwen3ASRConfig.from_dir(tmp_path).model_type == "qwen3_asr"
