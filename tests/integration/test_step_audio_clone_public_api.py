"""Manual local public-API smoke test for Step-Audio-EditX."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mlx_speech.generation.step_audio_editx import StepAudioEditXModel
from tests.helpers.step_audio import EDITX_DIR, PROMPT_AUDIO, TOKENIZER_DIR, skip_no_integration

pytestmark = pytest.mark.integration


@skip_no_integration
def test_step_audio_clone_public_api_produces_waveform() -> None:
    audio, sample_rate = sf.read(PROMPT_AUDIO, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)

    model = StepAudioEditXModel.from_dir(
        EDITX_DIR,
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
