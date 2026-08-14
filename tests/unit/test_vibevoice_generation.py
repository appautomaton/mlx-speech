"""Tests for VibeVoice generation loop."""

import mlx.core as mx

from mlx_speech.generation.vibevoice import (
    VibeVoiceGenerationConfig,
    _apply_top_p,
    _constrain_logits,
    _format_text_input,
    _sample_next_token,
)


class TestConstrainLogits:
    def test_masks_invalid_tokens(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        valid = [1, 3]
        result = _constrain_logits(logits, valid)
        mx.eval(result)
        # Only positions 1 and 3 should have finite values
        assert float(result[0, 0]) == float("-inf")
        assert float(result[0, 1]) == 2.0
        assert float(result[0, 2]) == float("-inf")
        assert float(result[0, 3]) == 4.0
        assert float(result[0, 4]) == float("-inf")

    def test_argmax_selects_valid(self):
        logits = mx.array([[10.0, 1.0, 20.0, 5.0]])
        valid = [1, 3]
        result = _constrain_logits(logits, valid)
        best = mx.argmax(result, axis=-1).item()
        assert best == 3  # highest among valid tokens


class TestGenerationConfig:
    def test_defaults(self):
        cfg = VibeVoiceGenerationConfig()
        assert cfg.max_new_tokens == 4096
        assert cfg.cfg_scale == 1.3
        assert cfg.diffusion_steps == 20
        assert cfg.do_sample is False
        assert cfg.top_p == 1.0
        assert cfg.seed is None


class TestSamplingHelpers:
    def test_top_p_masks_removed_logits(self):
        logits = mx.array([[4.0, 3.0, 2.0, 1.0]], dtype=mx.float32)

        filtered = _apply_top_p(logits, top_p=0.6)
        mx.eval(filtered)

        assert float(filtered[0, 0]) == 4.0
        assert any(float(filtered[0, i]) < -1e30 for i in range(1, 4))

    def test_sample_next_token_uses_seed_for_reproducible_sampling(self):
        logits = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float32)
        valid_ids = [0, 1, 2, 3]

        mx.random.seed(123)
        first = _sample_next_token(
            logits,
            valid_ids=valid_ids,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
        )

        mx.random.seed(123)
        second = _sample_next_token(
            logits,
            valid_ids=valid_ids,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
        )

        assert first.tolist() == second.tolist()

    def test_sample_next_token_greedy_ignores_temperature_and_top_p(self):
        logits = mx.array([[1.0, 2.0, 5.0, 3.0]], dtype=mx.float32)

        token = _sample_next_token(
            logits,
            valid_ids=[0, 1, 2, 3],
            temperature=0.0,
            top_p=0.2,
            do_sample=False,
        )

        assert token.tolist() == [2]


class TestPromptFormatting:
    def test_plain_text_defaults_to_speaker_zero(self):
        assert _format_text_input("Hello there.") == "Speaker 0: Hello there."

    def test_existing_speaker_labels_are_preserved(self):
        text = "Speaker 1: Hello.\nSpeaker 2: Hi."
        assert _format_text_input(text) == text

    def test_bracket_speaker_labels_are_treated_as_plain_text(self):
        text = "[1]: Hello.\n[2]: Hi."
        assert _format_text_input(text) == "Speaker 0: [1]: Hello. [2]: Hi."
