"""Manual local public-API smoke test for Step-Audio-EditX."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mlx_speech.generation.step_audio_editx import StepAudioEditXModel


MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
TOKENIZER_DIR = Path("models/stepfun/step_audio_tokenizer/original")
PROMPT_AUDIO = Path("outputs/source/hank_hill_ref.wav")

RUN_LOCAL_INTEGRATION = os.environ.get("RUN_LOCAL_INTEGRATION") == "1"
HAS_LOCAL_ASSETS = MODEL_DIR.exists() and TOKENIZER_DIR.exists() and PROMPT_AUDIO.exists()


@pytest.mark.skipif(
    not RUN_LOCAL_INTEGRATION or not HAS_LOCAL_ASSETS,
    reason="manual local integration test; requires RUN_LOCAL_INTEGRATION=1 and local Step-Audio assets",
)
def test_step_audio_clone_public_api_produces_waveform() -> None:
    audio, sample_rate = sf.read(PROMPT_AUDIO, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)

    model = StepAudioEditXModel.from_dir(
        MODEL_DIR,
        tokenizer_dir=TOKENIZER_DIR,
        prefer_mlx_int8=False,
    )
    result = model.clone(
        audio,
        sample_rate,
        "Loud is not allowed. Now you listen.",
        "Testing the local nonstream waveform pipeline.",
        max_new_tokens=32,
        temperature=0.0,
        flow_steps=2,
    )

    assert result.sample_rate == 24000
    assert result.mode == "clone"
    assert len(result.generated_token_ids) > 0
    assert result.waveform.ndim == 1
    assert result.waveform.shape[0] > 0
    assert result.waveform.dtype == np.float32
    assert np.isfinite(result.waveform).all()
    assert float(np.abs(result.waveform).sum()) > 0.0

