"""Checkpoint-loading test for the Gemma 3 text backbone.

Tier-2 test: requires the local `models/gemma_3_12b_it_backbone/mlx-4bit/` directory.
Skipped automatically if absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.gemma3_text import load_gemma3_text_model

GEMMA_DIR = Path("models/gemma_3_12b_it_backbone/mlx-4bit")

pytestmark = pytest.mark.skipif(
    not GEMMA_DIR.is_dir(),
    reason="Gemma checkpoint dir not present; skipping",
)


def test_load_gemma3_text_model_returns_model_and_config():
    model, config = load_gemma3_text_model(GEMMA_DIR)
    assert config.num_hidden_layers == 48
    assert config.hidden_size == 3840
    assert config.head_dim == 256
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.vocab_size == 262208
    # Sanity: model has 48 decoder layers
    assert len(model.layers) == 48


def test_load_gemma3_text_forward_one_token_does_not_nan():
    """End-to-end smoke: load the 4-bit checkpoint, run a 4-token forward
    pass, assert no NaN/Inf in any of the 49 hidden states."""
    model, config = load_gemma3_text_model(GEMMA_DIR)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    attention_mask = mx.ones((1, 4), dtype=mx.int32)
    out = model(input_ids, attention_mask)

    assert len(out.hidden_states) == 49
    for i, h in enumerate(out.hidden_states):
        assert h.shape == (1, 4, config.hidden_size)
        assert mx.all(mx.isfinite(h)).item(), f"non-finite values in hidden_states[{i}]"
    assert mx.all(mx.isfinite(out.last_hidden_state)).item()
