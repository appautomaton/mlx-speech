"""Runtime smoke for local Granite Speech ASR."""

from __future__ import annotations

from pathlib import Path

import pytest


GRANITE_DIR = Path("models/ibm/granite_4_0_1b_speech/original")
GRANITE_INT8_DIR = Path("models/ibm/granite_4_0_1b_speech/mlx-int8")


_CASES = (
    pytest.param(
        GRANITE_DIR,
        id="original-bf16",
        marks=pytest.mark.skipif(
            not (GRANITE_DIR / "model.safetensors.index.json").exists(),
            reason="Granite Speech original checkpoint missing",
        ),
    ),
    pytest.param(
        GRANITE_INT8_DIR,
        id="mlx-int8",
        marks=pytest.mark.skipif(
            not (GRANITE_INT8_DIR / "model.safetensors").exists(),
            reason="Granite Speech int8 artifact missing",
        ),
    ),
)


@pytest.mark.runtime
@pytest.mark.parametrize("model_dir", _CASES)
def test_granite_speech_smoke_transcribes_multilingual_sample(model_dir: Path):
    from mlx_speech import asr

    sample = model_dir / "multilingual_sample.wav"
    if not sample.exists():
        pytest.skip("Granite Speech sample audio missing")
    model = asr.load(str(model_dir))
    result = model.generate(sample, max_new_tokens=32)

    assert result.text.strip()
    assert "timothy was a spoiled cat" in result.text.lower()
    assert result.language == "en"
