"""Reference prompts require a paired transcript, and the encoder pads causally."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.generation.breeze_tts import _prompt_text, _validated_reference
from mlx_speech.models.breeze_tts.audio_encoder import _conv_padding


def test_reference_audio_and_transcript_are_a_pair() -> None:
    with pytest.raises(ValueError, match="together"):
        _validated_reference(mx.zeros((4,)), None)
    with pytest.raises(ValueError, match="together"):
        _validated_reference(None, "hello")
    assert _validated_reference(None, "  ") is None
    audio, text = _validated_reference(mx.zeros((4,)), " hello ")
    assert text == "hello"
    assert audio.shape == (4,)


def test_clone_prompt_puts_instruction_only_on_the_target() -> None:
    assert _prompt_text("S0", "target", None) == "[S0]target"
    assert _prompt_text("S0", "target", "Speak slowly.") == (
        "[S0]<ins_bos>Speak slowly.<ins_eos>target"
    )


def test_causal_encoder_conv_keeps_the_first_frame_aligned() -> None:
    assert _conv_padding(100, 7, 1, 1, causal=True) == (6, 0)
    assert _conv_padding(100, 8, 4, 1, causal=True) == (4, 0)
