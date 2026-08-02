from __future__ import annotations

import pytest

from mlx_speech.models.dots_tts.config import DotsTTSConfig, DotsTTSQwenConfig


def _transformer(*, input_dim: int | None = None) -> dict:
    payload = {
        "num_layers": 2,
        "num_heads": 4,
        "hidden_size": 16,
        "ffn_hidden_size": 32,
        "modulation": False,
        "qkv_bias": False,
        "qk_norm": True,
        "attn_dropout": 0.0,
        "dropout": 0.0,
        "norm_layer": "RMSNorm",
        "alibi_bias": False,
        "rotary_bias": True,
        "rotary_theta": 10_000.0,
    }
    if input_dim is not None:
        payload.update({"input_dim": input_dim, "causal": True})
    return payload


def _vocoder() -> dict:
    return {
        "sample_rate": 48_000,
        "upsample_rates": [2, 2],
        "upsample_kernel_sizes": [4, 4],
        "upsample_initial_channel": 16,
        "resblock": "1",
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "downsample_rates": [2, 2],
        "downsample_channels": [4, 8, 16],
        "activation": "snakebeta",
        "snake_logscale": True,
        "latent_dim": 128,
        "causal": True,
        "mi_num_layers": 1,
        "causal_encoder": True,
        "use_bias_at_final": False,
        "use_tanh_at_final": False,
    }


def dots_config(*, meanflow: bool = False) -> dict:
    payload = {
        "model_type": "dots_tts",
        "latent_dim": 128,
        "patch_size": 4,
        "cfg_droprate": 0.2,
        "PatchEncoder": _transformer(input_dim=128),
        "DiT": _transformer(),
        "vocoder": _vocoder(),
        "fm_sigma": 0.0,
        "campplus_embedding_size": 512,
        "xvec_max_audio_seconds": 10.0,
    }
    if meanflow:
        payload["meanflow"] = {"enabled": True, "use_duration_embedding": True}
    return payload


def qwen_config() -> dict:
    return {
        "model_type": "qwen2",
        "vocab_size": 151_672,
        "max_position_embeddings": 131_072,
        "hidden_size": 1_536,
        "intermediate_size": 8_960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }


def test_config_distinguishes_soar_and_meanflow() -> None:
    soar = DotsTTSConfig.from_dict(dots_config())
    meanflow = DotsTTSConfig.from_dict(dots_config(meanflow=True))
    assert soar.mode == "flow_matching"
    assert meanflow.mode == "meanflow"
    assert soar.vocoder.hop_size == 4


def test_config_preserves_unknown_upstream_fields() -> None:
    payload = dots_config()
    payload["future_top_level"] = {"enabled": True}
    payload["DiT"]["future_attention"] = "kept"
    config = DotsTTSConfig.from_dict(payload)
    assert config.to_dict()["future_top_level"] == {"enabled": True}
    assert config.to_dict()["DiT"]["future_attention"] == "kept"


def test_config_rejects_inconsistent_meanflow_and_vocoder() -> None:
    payload = dots_config(meanflow=True)
    payload["meanflow"]["use_duration_embedding"] = False
    with pytest.raises(ValueError, match="duration embeddings"):
        DotsTTSConfig.from_dict(payload)
    payload = dots_config()
    payload["vocoder"]["latent_dim"] = 64
    with pytest.raises(ValueError, match="latent dimensions differ"):
        DotsTTSConfig.from_dict(payload)


def test_qwen_config_validates_gqa_and_preserves_unknown_fields() -> None:
    payload = qwen_config()
    payload["transformers_version"] = "4.57.0"
    config = DotsTTSQwenConfig.from_dict(payload)
    assert config.num_attention_heads == 12
    assert config.num_key_value_heads == 2
    assert config.to_dict()["transformers_version"] == "4.57.0"
    payload["num_key_value_heads"] = 5
    with pytest.raises(ValueError, match="divide evenly"):
        DotsTTSQwenConfig.from_dict(payload)
