"""Tokenizer behavior tests for the plain-text Gemma wrapper used by DramaBox.

Tests run against the real `models/gemma_3_12b_it_backbone/mlx-4bit/tokenizer.json`; they
skip if the model directory is not present locally.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.gemma3_text import LTXVGemmaTokenizer

GEMMA_DIR = Path("models/gemma_3_12b_it_backbone/mlx-4bit")

pytestmark = pytest.mark.skipif(
    not (GEMMA_DIR / "tokenizer.json").is_file(),
    reason="Gemma tokenizer.json not present; skipping (requires local model dir)",
)


def test_from_dir_loads():
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    assert tok.pad_token_id is not None


def test_encode_returns_left_padded_shape():
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    input_ids, attention_mask = tok.encode("hello", max_length=64)
    assert input_ids.shape == (1, 64)
    assert attention_mask.shape == (1, 64)
    # Left padding: trailing positions are real (mask=1); leading are pad (mask=0)
    mask_list = attention_mask[0].tolist()
    # The last few positions must be 1 (real tokens land on the right edge)
    assert mask_list[-1] == 1
    # The first position must be 0 if any padding occurred
    assert mask_list[0] == 0


def test_encode_long_input_truncates_from_right():
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    long_text = "word " * 5000  # certainly more than max_length=64 tokens
    input_ids, attention_mask = tok.encode(long_text, max_length=64)
    assert input_ids.shape == (1, 64)
    # All positions should be valid (no padding since we filled the window)
    assert attention_mask.sum().item() == 64


def test_encode_strips_whitespace():
    """DramaBox tokenizer strips leading/trailing whitespace before encoding."""
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    ids_a, _ = tok.encode("hello world", max_length=32)
    ids_b, _ = tok.encode("   hello world   ", max_length=32)
    assert (ids_a == ids_b).all().item()


def test_encode_batch_uniform_length():
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    ids, mask = tok.encode_batch(["short", "a bit longer sentence here"], max_length=32)
    assert ids.shape == (2, 32)
    assert mask.shape == (2, 32)
    # First row has more padding (shorter input)
    assert mask[0].sum().item() < mask[1].sum().item()


def test_encode_dtype_is_int32():
    tok = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)
    input_ids, attention_mask = tok.encode("hello", max_length=16)
    assert input_ids.dtype == mx.int32
    assert attention_mask.dtype == mx.int32
