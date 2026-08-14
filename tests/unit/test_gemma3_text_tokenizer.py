"""Tokenizer behavior tests for the plain-text Gemma wrapper used by DramaBox."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from mlx_speech.models.gemma3_text import LTXVGemmaTokenizer


@pytest.fixture()
def gemma_dir(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "hello": 3,
                "world": 4,
                "word": 5,
                "short": 6,
                "a": 7,
                "bit": 8,
                "longer": 9,
                "sentence": 10,
                "here": 11,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "<pad>", "eos_token": "<eos>"}),
        encoding="utf-8",
    )
    return tmp_path


def test_from_dir_loads(gemma_dir: Path):
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    assert tok.pad_token_id is not None


def test_encode_returns_left_padded_shape(gemma_dir: Path):
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    input_ids, attention_mask = tok.encode("hello", max_length=64)
    assert input_ids.shape == (1, 64)
    assert attention_mask.shape == (1, 64)
    # Left padding: trailing positions are real (mask=1); leading are pad (mask=0)
    mask_list = attention_mask[0].tolist()
    # The last few positions must be 1 (real tokens land on the right edge)
    assert mask_list[-1] == 1
    # The first position must be 0 if any padding occurred
    assert mask_list[0] == 0


def test_encode_long_input_truncates_from_right(gemma_dir: Path):
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    long_text = "word " * 5000  # certainly more than max_length=64 tokens
    input_ids, attention_mask = tok.encode(long_text, max_length=64)
    assert input_ids.shape == (1, 64)
    # All positions should be valid (no padding since we filled the window)
    assert attention_mask.sum().item() == 64


def test_encode_strips_whitespace(gemma_dir: Path):
    """DramaBox tokenizer strips leading/trailing whitespace before encoding."""
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    ids_a, _ = tok.encode("hello world", max_length=32)
    ids_b, _ = tok.encode("   hello world   ", max_length=32)
    assert (ids_a == ids_b).all().item()


def test_encode_batch_uniform_length(gemma_dir: Path):
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    ids, mask = tok.encode_batch(["short", "a bit longer sentence here"], max_length=32)
    assert ids.shape == (2, 32)
    assert mask.shape == (2, 32)
    # First row has more padding (shorter input)
    assert mask[0].sum().item() < mask[1].sum().item()


def test_encode_dtype_is_int32(gemma_dir: Path):
    tok = LTXVGemmaTokenizer.from_dir(gemma_dir)
    input_ids, attention_mask = tok.encode("hello", max_length=16)
    assert input_ids.dtype == mx.int32
    assert attention_mask.dtype == mx.int32
