from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.moss_delay import (
    build_sound_effect_conversation,
    estimate_sound_effect_tokens,
    resolve_moss_sound_effect_model_dir,
)


class _DummySoundEffectProcessor:
    def build_user_message(self, *, ambient_sound: str, tokens: int):
        return {
            "role": "user",
            "content": f"- Ambient Sound:\n{ambient_sound}\n- Tokens:\n{tokens}\n",
        }


def test_estimate_sound_effect_tokens_matches_upstream_heuristic() -> None:
    assert estimate_sound_effect_tokens(1.0) == 12
    assert estimate_sound_effect_tokens(10.0) == 125


def test_build_sound_effect_conversation_uses_ambient_sound_and_expected_tokens() -> None:
    processor = _DummySoundEffectProcessor()

    conversations, expected_tokens = build_sound_effect_conversation(
        processor,
        ambient_sound="a sports car roaring past on the highway.",
        duration_seconds=10.0,
    )

    assert expected_tokens == 125
    assert len(conversations) == 1
    assert conversations[0][0]["role"] == "user"
    assert "Ambient Sound" in conversations[0][0]["content"]


def test_build_sound_effect_conversation_rejects_empty_prompt() -> None:
    processor = _DummySoundEffectProcessor()

    with pytest.raises(ValueError, match="ambient_sound"):
        build_sound_effect_conversation(
            processor,
            ambient_sound="",
            duration_seconds=10.0,
        )


def test_resolve_moss_sound_effect_model_dir_accepts_explicit_override() -> None:
    custom = Path("/tmp/custom-sound-effect")

    assert resolve_moss_sound_effect_model_dir(custom) == custom
