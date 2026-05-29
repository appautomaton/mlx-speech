from __future__ import annotations

import json

from mlx_speech.models.granite_speech_asr.config import GraniteSpeechConfig


def _granite_config_payload() -> dict:
    return {
        "audio_token_index": 100352,
        "downsample_rate": 5,
        "encoder_config": {
            "context_size": 200,
            "conv_expansion_factor": 2,
            "conv_kernel_size": 15,
            "dim_head": 128,
            "dropout": 0.1,
            "feedforward_mult": 4,
            "hidden_dim": 1024,
            "input_dim": 160,
            "max_pos_emb": 512,
            "model_type": "granite_speech_encoder",
            "num_heads": 8,
            "num_layers": 16,
            "output_dim": 348,
            "torch_dtype": "bfloat16",
        },
        "has_lora_adapter": False,
        "initializer_range": 0.02,
        "model_type": "granite_speech",
        "projector_config": {
            "attention_probs_dropout_prob": 0.1,
            "cross_attention_frequency": 1,
            "encoder_hidden_size": 1024,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 2048,
            "model_type": "blip_2_qformer",
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "position_embedding_type": "absolute",
            "torch_dtype": "bfloat16",
            "use_qformer_text_input": False,
            "vocab_size": 30522,
        },
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_multiplier": 0.0078125,
            "bos_token_id": 100257,
            "dtype": "float32",
            "embedding_multiplier": 12.0,
            "eos_token_id": 100257,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.1,
            "intermediate_size": 4096,
            "logits_scaling": 8.0,
            "max_position_embeddings": 4096,
            "mlp_bias": False,
            "model_type": "granite",
            "num_attention_heads": 16,
            "num_hidden_layers": 40,
            "num_key_value_heads": 4,
            "pad_token_id": 100256,
            "residual_multiplier": 0.22,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 100353,
        },
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "window_size": 15,
    }


def test_granite_speech_config_parses_upstream_shape(tmp_path):
    model_dir = tmp_path / "granite"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_granite_config_payload()), encoding="utf-8")

    cfg = GraniteSpeechConfig.from_path(model_dir)

    assert cfg.model_type == "granite_speech"
    assert cfg.audio_token_index == 100352
    assert cfg.window_size == 15
    assert cfg.downsample_rate == 5
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.encoder.input_dim == 160
    assert cfg.encoder.hidden_dim == 1024
    assert cfg.encoder.num_layers == 16
    assert cfg.encoder.torch_dtype == "bfloat16"
    assert cfg.projector.hidden_size == 1024
    assert cfg.projector.encoder_hidden_size == 1024
    assert cfg.projector.torch_dtype == "bfloat16"
    assert cfg.text.hidden_size == 2048
    assert cfg.text.num_hidden_layers == 40
    assert cfg.text.dtype == "float32"
    assert cfg.text.torch_dtype == "bfloat16"
    assert cfg.text.embedding_multiplier == 12.0
    assert cfg.text.logits_scaling == 8.0


def test_granite_speech_config_round_trips_extra_fields():
    payload = _granite_config_payload()
    payload["transformers_version"] = "4.54.0"
    payload["projector_config"]["_attn_implementation_autoset"] = True

    cfg = GraniteSpeechConfig.from_dict(payload)
    round_trip = cfg.to_dict()

    assert round_trip["transformers_version"] == "4.54.0"
    assert round_trip["projector_config"]["_attn_implementation_autoset"] is True
    assert round_trip["text_config"]["rope_parameters"] == {
        "rope_theta": 10000,
        "rope_type": "default",
    }
