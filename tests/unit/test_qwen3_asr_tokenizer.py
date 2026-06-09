from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from mlx_speech.models.qwen3_asr import Qwen3ASRTokenizer
import mlx_speech.models.qwen3_asr.tokenizer as tokenizer_module


@dataclass(frozen=True)
class FakeEncoding:
    ids: list[int]


class FakeBPETokenizer:
    _ids = {
        "<|audio_pad|>": 151676,
        "<|audio_start|>": 151669,
        "<|audio_end|>": 151670,
        "<|endoftext|>": 151645,
        "<|pad|>": 151643,
    }

    def token_to_id(self, token: str) -> int | None:
        return self._ids.get(token)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> FakeEncoding:
        assert add_special_tokens is False
        return FakeEncoding([ord(char) for char in text])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def _write_config(
    model_dir,
    *,
    audio_ids: tuple[int, int, int] = (151676, 151669, 151670),
):
    audio_token_id, audio_start_token_id, audio_end_token_id = audio_ids
    payload = {
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_token_id": audio_token_id,
            "audio_start_token_id": audio_start_token_id,
            "audio_end_token_id": audio_end_token_id,
            "audio_config": {
                "d_model": 1,
                "num_mel_bins": 1,
                "encoder_layers": 1,
                "encoder_attention_heads": 1,
                "encoder_ffn_dim": 1,
                "downsample_hidden_size": 1,
                "output_dim": 1,
                "max_source_positions": 1,
                "n_window": 1,
                "n_window_infer": 1,
                "conv_chunksize": 1,
            },
            "text_config": {
                "hidden_size": 1,
                "intermediate_size": 1,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 1,
                "vocab_size": 20,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
            },
        },
    }
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_tokenizer_assets(
    model_dir,
    *,
    audio_ids: tuple[int, int, int] = (151676, 151669, 151670),
):
    _write_config(model_dir, audio_ids=audio_ids)
    tokenizer_config = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "model_max_length": 4096,
    }
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


def _patch_fake_tokenizer(monkeypatch):
    def fake_load_bpe_tokenizer(**_kwargs):
        return FakeBPETokenizer()

    monkeypatch.setattr(tokenizer_module, "_load_bpe_tokenizer", fake_load_bpe_tokenizer)


def test_qwen3_asr_tokenizer_resolves_audio_token_ids(monkeypatch, tmp_path):
    _write_tokenizer_assets(tmp_path)
    _patch_fake_tokenizer(monkeypatch)

    tokenizer = Qwen3ASRTokenizer.from_dir(tmp_path)

    assert tokenizer.audio_token == "<|audio_pad|>"
    assert tokenizer.audio_bos_token == "<|audio_start|>"
    assert tokenizer.audio_eos_token == "<|audio_end|>"
    assert tokenizer.audio_token_id == 151676
    assert tokenizer.audio_bos_token_id == 151669
    assert tokenizer.audio_eos_token_id == 151670
    assert tokenizer.token_to_id("<|audio_pad|>") == 151676
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.pad_token_id == 151643
    assert tokenizer.model_max_length == 4096


def test_qwen3_asr_tokenizer_roundtrips_english_and_chinese(monkeypatch, tmp_path):
    _write_tokenizer_assets(tmp_path)
    _patch_fake_tokenizer(monkeypatch)

    tokenizer = Qwen3ASRTokenizer.from_dir(tmp_path)

    for text in ("hello world", "你好，world"):
        token_ids = tokenizer.encode(text)
        assert token_ids
        assert tokenizer.decode(token_ids) == text


def test_qwen3_asr_tokenizer_rejects_audio_token_mismatch(monkeypatch, tmp_path):
    _write_tokenizer_assets(tmp_path, audio_ids=(99, 151669, 151670))
    _patch_fake_tokenizer(monkeypatch)

    with pytest.raises(ValueError, match="audio_token_id mismatch"):
        Qwen3ASRTokenizer.from_dir(tmp_path)


def test_qwen3_asr_tokenizer_reports_missing_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="config"):
        Qwen3ASRTokenizer.from_dir(tmp_path)
