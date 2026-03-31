"""Helpers for MOSS-SoundEffect inference on top of MossTTSDelay."""

from __future__ import annotations

from .processor import MossTTSDelayProcessor


SOUND_EFFECT_TOKENS_PER_SECOND = 12.5
SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS = 4096
SOUND_EFFECT_DEFAULT_AUDIO_TEMPERATURE = 1.5
SOUND_EFFECT_DEFAULT_AUDIO_TOP_P = 0.6
SOUND_EFFECT_DEFAULT_AUDIO_TOP_K = 50
SOUND_EFFECT_DEFAULT_AUDIO_REPETITION_PENALTY = 1.2


def estimate_sound_effect_tokens(duration_seconds: float) -> int:
    """Convert target duration to the upstream expected-token heuristic."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    return max(1, int(float(duration_seconds) * SOUND_EFFECT_TOKENS_PER_SECOND))


def build_sound_effect_conversation(
    processor: MossTTSDelayProcessor,
    *,
    ambient_sound: str,
    duration_seconds: float | None = None,
    expected_tokens: int | None = None,
) -> tuple[list[list[dict[str, object]]], int]:
    """Build a single-sample MOSS-SoundEffect conversation."""

    ambient_sound_text = (ambient_sound or "").strip()
    if not ambient_sound_text:
        raise ValueError("ambient_sound must be a non-empty description.")

    if expected_tokens is None:
        if duration_seconds is None:
            raise ValueError("Provide either duration_seconds or expected_tokens.")
        resolved_tokens = estimate_sound_effect_tokens(duration_seconds)
    else:
        resolved_tokens = max(1, int(expected_tokens))

    return [[processor.build_user_message(ambient_sound=ambient_sound_text, tokens=resolved_tokens)]], resolved_tokens
