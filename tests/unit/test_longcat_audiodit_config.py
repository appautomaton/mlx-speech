from __future__ import annotations

import json
from pathlib import Path

from mlx_speech.models.longcat_audiodit.config import (
    LongCatAudioDiTConfig,
    LongCatTextEncoderConfig,
    LongCatVaeConfig,
    QuantizationConfig,
)


def _sample_config_payload() -> dict:
    return {
        "model_type": "audiodit",
        "sampling_rate": 24000,
        "latent_hop": 2048,
        "latent_dim": 64,
        "max_wav_duration": 60,
        "dit_dim": 2560,
        "dit_depth": 32,
        "dit_heads": 32,
        "dit_ff_mult": 3.6,
        "dit_bias": True,
        "dit_eps": 1e-6,
        "dit_dropout": 0.0,
        "dit_adaln_type": "global",
        "dit_adaln_use_text_cond": True,
        "dit_cross_attn": True,
        "dit_cross_attn_norm": False,
        "dit_long_skip": True,
        "dit_qk_norm": True,
        "dit_text_conv": True,
        "dit_text_dim": 768,
        "dit_use_latent_condition": True,
        "repa_dit_layer": 8,
        "sigma": 0.0,
        "text_encoder_model": "google/umt5-base",
        "text_norm_feat": True,
        "text_add_embed": True,
        "text_encoder_config": {
            "_name_or_path": "ArthurZ/umt5-base",
            "model_type": "umt5",
            "d_model": 768,
            "d_ff": 2048,
            "d_kv": 64,
            "num_layers": 12,
            "num_decoder_layers": 12,
            "num_heads": 12,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "is_gated_act": True,
            "dense_act_fn": "gelu_new",
            "vocab_size": 256384,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "decoder_start_token_id": 0,
        },
        "vae_config": {
            "model_type": "audiodit_vae",
            "in_channels": 1,
            "channels": 128,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8],
            "latent_dim": 64,
            "encoder_latent_dim": 128,
            "use_snake": True,
            "downsample_shortcut": "averaging",
            "upsample_shortcut": "duplicating",
            "out_shortcut": "averaging",
            "in_shortcut": "duplicating",
            "final_tanh": False,
            "downsampling_ratio": 2048,
            "sample_rate": 24000,
            "scale": 0.71,
        },
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "transformers_version": "5.3.0",
    }


def test_config_from_dict_reads_nested_longcat_sections() -> None:
    cfg = LongCatAudioDiTConfig.from_dict(_sample_config_payload())

    assert cfg.model_type == "audiodit"
    assert cfg.dit_dim == 2560
    assert cfg.text_encoder_model == "google/umt5-base"
    assert cfg.quantization == QuantizationConfig(bits=8, group_size=64, mode="affine")

    assert isinstance(cfg.text_encoder_config, LongCatTextEncoderConfig)
    assert cfg.text_encoder_config.d_model == 768
    assert cfg.text_encoder_config.num_layers == 12
    assert cfg.text_encoder_config.extra["_name_or_path"] == "ArthurZ/umt5-base"
    assert cfg.text_encoder_config.extra["dense_act_fn"] == "gelu_new"

    assert isinstance(cfg.vae_config, LongCatVaeConfig)
    assert cfg.vae_config.c_mults == (1, 2, 4, 8, 16)
    assert cfg.vae_config.strides == (2, 4, 4, 8, 8)
    assert cfg.vae_config.scale == 0.71
    assert cfg.vae_config.extra["model_type"] == "audiodit_vae"

    assert cfg.extra["transformers_version"] == "5.3.0"


def test_config_from_path_reads_json_file(tmp_path: Path) -> None:
    payload = _sample_config_payload()
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg = LongCatAudioDiTConfig.from_path(tmp_path)

    assert cfg.sampling_rate == 24000
    assert cfg.latent_hop == 2048
    assert cfg.text_encoder_config.vocab_size == 256384
    assert cfg.vae_config.downsampling_ratio == 2048


def test_config_to_dict_round_trips_nested_extra_fields() -> None:
    cfg = LongCatAudioDiTConfig.from_dict(_sample_config_payload())

    payload = cfg.to_dict()

    assert payload["text_encoder_config"]["_name_or_path"] == "ArthurZ/umt5-base"
    assert payload["vae_config"]["model_type"] == "audiodit_vae"
    assert payload["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}
    assert payload["transformers_version"] == "5.3.0"
