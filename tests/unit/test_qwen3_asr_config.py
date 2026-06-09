from __future__ import annotations

import json

import pytest

from mlx_speech.models.qwen3_asr import Qwen3ASRConfig


def _write_qwen3_asr_config(model_dir):
    payload = {
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "model_type": "qwen3_asr",
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_config": {
                "model_type": "qwen3_asr_audio_encoder",
                "num_mel_bins": 128,
                "encoder_layers": 24,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "d_model": 1024,
                "downsample_hidden_size": 480,
                "output_dim": 2048,
                "max_source_positions": 1500,
                "n_window": 100,
                "n_window_infer": 3000,
                "conv_chunksize": 8,
            },
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 2048,
                "intermediate_size": 11008,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
            },
        },
    }
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_qwen3_asr_config_reads_1_7b_dimensions_from_fixture(tmp_path):
    _write_qwen3_asr_config(tmp_path)
    config = Qwen3ASRConfig.from_dir(tmp_path)

    assert config.model_type == "qwen3_asr"
    assert config.architectures == ("Qwen3ASRForConditionalGeneration",)
    assert "English" in config.support_languages
    assert "Chinese" in config.support_languages

    assert config.audio_token_id == 151676
    assert config.audio_start_token_id == 151669
    assert config.audio_end_token_id == 151670

    audio = config.audio_config
    assert audio.num_mel_bins == 128
    assert audio.encoder_layers == 24
    assert audio.encoder_attention_heads == 16
    assert audio.d_model == 1024
    assert audio.downsample_hidden_size == 480
    assert audio.output_dim == 2048

    text = config.text_config
    assert text.model_type == "qwen3"
    assert text.hidden_size == 2048
    assert text.num_hidden_layers == 28
    assert text.num_attention_heads == 16
    assert text.num_key_value_heads == 8
    assert text.vocab_size == 151936


def test_qwen3_asr_config_rejects_wrong_model_type(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "not_qwen"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="qwen3_asr"):
        Qwen3ASRConfig.from_dir(tmp_path)


def test_qwen3_asr_config_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config"):
        Qwen3ASRConfig.from_dir(tmp_path)
