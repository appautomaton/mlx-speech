"""Tests for VibeVoice checkpoint loading."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_voice.models.vibevoice.checkpoint import (
    load_vibevoice_checkpoint,
    load_vibevoice_model,
    sanitize_state_dict,
)

MODEL_DIR = Path("models/vibevoice/mlx-int8")
HAS_CHECKPOINT = any(MODEL_DIR.glob("*.safetensors")) if MODEL_DIR.exists() else False


class TestSanitize:
    def test_conv1d_transpose(self):
        """Conv1d weights should be transposed from (out, in, k) to (out, k, in)."""
        weights = {
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight": mx.zeros((16, 32, 7)),
            "model.acoustic_tokenizer.decoder.head.conv.conv.bias": mx.zeros((16,)),
        }
        sanitized, _, _ = sanitize_state_dict(weights)
        w = sanitized["model.acoustic_tokenizer.decoder.head.conv.conv.weight"]
        assert w.shape == (16, 7, 32)  # (out, k, in)

    def test_convtr_transpose(self):
        """ConvTranspose1d weights: (in, out, k) → (out, k, in)."""
        weights = {
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight": mx.zeros((64, 32, 16)),
        }
        sanitized, _, _ = sanitize_state_dict(weights)
        key = "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight"
        w = sanitized[key]
        assert w.shape == (32, 16, 64)  # (out, k, in)

    def test_linear_untouched(self):
        weights = {"lm_head.weight": mx.zeros((152064, 3584))}
        sanitized, _, _ = sanitize_state_dict(weights)
        assert sanitized["lm_head.weight"].shape == (152064, 3584)

    def test_skip_inv_freq(self):
        weights = {
            "model.language_model.rotary_emb.inv_freq": mx.zeros((64,)),
            "lm_head.weight": mx.zeros((10, 5)),
        }
        sanitized, skipped, _ = sanitize_state_dict(weights)
        assert "model.language_model.rotary_emb.inv_freq" in skipped
        assert "model.language_model.rotary_emb.inv_freq" not in sanitized


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint not available")
class TestRealCheckpoint:
    def test_load_checkpoint(self):
        ckpt = load_vibevoice_checkpoint(MODEL_DIR)
        assert ckpt.key_count > 0
        assert ckpt.config.model_type == "vibevoice"

    def test_model_alignment(self):
        loaded = load_vibevoice_model(MODEL_DIR, strict=False)
        assert loaded.alignment_report.is_exact_match
