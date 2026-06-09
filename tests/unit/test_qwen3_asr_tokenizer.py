from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.qwen3_asr import Qwen3ASRTokenizer


QWEN_DIR = Path("models/qwen3_asr_1_7b/original")
TOKENIZER_FILES = (
    QWEN_DIR / "config.json",
    QWEN_DIR / "tokenizer_config.json",
    QWEN_DIR / "vocab.json",
    QWEN_DIR / "merges.txt",
)


pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in TOKENIZER_FILES),
    reason="Qwen3-ASR tokenizer assets not present",
)


def test_qwen3_asr_tokenizer_resolves_audio_token_ids():
    tokenizer = Qwen3ASRTokenizer.from_dir(QWEN_DIR)

    assert tokenizer.audio_token == "<|audio_pad|>"
    assert tokenizer.audio_bos_token == "<|audio_start|>"
    assert tokenizer.audio_eos_token == "<|audio_end|>"
    assert tokenizer.audio_token_id == 151676
    assert tokenizer.audio_bos_token_id == 151669
    assert tokenizer.audio_eos_token_id == 151670
    assert tokenizer.token_to_id("<|audio_pad|>") == 151676
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.pad_token_id == 151643


def test_qwen3_asr_tokenizer_roundtrips_english_and_chinese():
    tokenizer = Qwen3ASRTokenizer.from_dir(QWEN_DIR)

    for text in ("hello world", "你好，world"):
        token_ids = tokenizer.encode(text)
        assert token_ids
        assert tokenizer.decode(token_ids) == text


def test_qwen3_asr_tokenizer_reports_missing_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="config"):
        Qwen3ASRTokenizer.from_dir(tmp_path)
