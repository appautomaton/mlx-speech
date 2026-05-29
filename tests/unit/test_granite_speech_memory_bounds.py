from __future__ import annotations

import inspect

import pytest

from mlx_speech.generation.granite_speech_asr import validate_context_window
import mlx_speech.diagnostics as diagnostics


def test_granite_context_window_allows_exact_fit():
    validate_context_window(
        prompt_tokens=4,
        max_new_tokens=4,
        max_position_embeddings=8,
    )


def test_granite_context_window_rejects_overflow():
    with pytest.raises(ValueError, match="prompt_tokens=7"):
        validate_context_window(
            prompt_tokens=7,
            max_new_tokens=2,
            max_position_embeddings=8,
        )


def test_granite_context_window_rejects_empty_generation():
    with pytest.raises(ValueError, match="positive"):
        validate_context_window(
            prompt_tokens=4,
            max_new_tokens=0,
            max_position_embeddings=8,
        )


def test_granite_memory_telemetry_has_no_background_polling():
    source = inspect.getsource(diagnostics)

    assert "threading" not in source
    assert "multiprocessing" not in source
    assert "time.sleep" not in source
    assert "while True" not in source
