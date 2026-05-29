"""Runtime smoke for local Granite Speech ASR."""

from __future__ import annotations

from pathlib import Path

import pytest


GRANITE_DIR = Path("models/ibm/granite_4_0_1b_speech/original")
SAMPLE = GRANITE_DIR / "multilingual_sample.wav"

pytestmark = pytest.mark.skipif(
    not (GRANITE_DIR / "model.safetensors.index.json").exists() or not SAMPLE.exists(),
    reason="Granite Speech checkpoint or sample audio missing",
)


@pytest.mark.runtime
def test_granite_speech_smoke_transcribes_multilingual_sample():
    from mlx_speech.generation.granite_speech_asr import GraniteSpeechAsrModel

    runtime = GraniteSpeechAsrModel.from_dir(GRANITE_DIR)
    result = runtime.transcribe(SAMPLE, max_new_tokens=32)

    assert result.text.strip()
    assert "timothy was a spoiled cat" in result.text.lower()
    assert result.tokens
