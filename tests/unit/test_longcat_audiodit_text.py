from __future__ import annotations

import pytest

from mlx_speech.models.longcat_audiodit.text import (
    approx_duration_from_text,
    normalize_text,
)


def test_normalize_text_matches_upstream_behavior_exactly() -> None:
    assert normalize_text("  Hello  “World”  ") == " hello world "
    assert normalize_text("A  B\nC") == "a b c"


def test_approx_duration_uses_language_specific_rates() -> None:
    assert approx_duration_from_text("hi") == pytest.approx(0.164)
    assert approx_duration_from_text("今天晴天") == pytest.approx(0.84)
    assert approx_duration_from_text("a!", max_duration=30.0) == pytest.approx(0.164)


def test_approx_duration_is_clamped_to_max_duration() -> None:
    assert approx_duration_from_text("z" * 1000, max_duration=3.0) == 3.0
