"""Tests for VibeVoice checkpoint loading."""

import mlx.core as mx

from mlx_speech.models.vibevoice.checkpoint import sanitize_state_dict


class TestSanitize:
    def test_conv1d_transpose(self):
        """Conv1d weights should be transposed from (out, in, k) to (out, k, in)."""
        weights = {
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight": mx.zeros(
                (16, 32, 7)
            ),
            "model.acoustic_tokenizer.decoder.head.conv.conv.bias": mx.zeros((16,)),
        }
        sanitized, _, _ = sanitize_state_dict(weights)
        w = sanitized["model.acoustic_tokenizer.decoder.head.conv.conv.weight"]
        assert w.shape == (16, 7, 32)  # (out, k, in)

    def test_convtr_transpose(self):
        """ConvTranspose1d weights: (in, out, k) → (out, k, in)."""
        weights = {
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight": mx.zeros(
                (64, 32, 16)
            ),
        }
        sanitized, _, _ = sanitize_state_dict(weights)
        key = (
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight"
        )
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
