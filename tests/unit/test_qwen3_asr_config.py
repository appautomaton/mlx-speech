from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_speech.models.qwen3_asr import Qwen3ASRConfig


QWEN_DIR = Path("models/qwen3_asr_1_7b/original")


pytestmark = pytest.mark.skipif(
    not (QWEN_DIR / "config.json").exists(),
    reason="Qwen3-ASR config assets not present",
)


def test_qwen3_asr_config_reads_real_1_7b_dimensions():
    config = Qwen3ASRConfig.from_dir(QWEN_DIR)

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
