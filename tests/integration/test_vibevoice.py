"""Manual end-to-end VibeVoice waveform coverage."""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.audio.io import load_audio
from mlx_speech.generation.vibevoice import (
    VibeVoiceGenerationConfig,
    VibeVoiceSynthesisOutput,
    synthesize_vibevoice,
)
from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model
from mlx_speech.models.vibevoice.tokenizer import VibeVoiceTokenizer


INT8_DIR = Path("models/vibevoice/mlx-int8")
ORIGINAL_DIR = Path("models/vibevoice/original")
HAS_INT8 = INT8_DIR.is_dir() and any(INT8_DIR.glob("*.safetensors"))
HAS_ORIGINAL = ORIGINAL_DIR.is_dir() and any(ORIGINAL_DIR.glob("*.safetensors"))
MODEL_DIR = INT8_DIR if HAS_INT8 else ORIGINAL_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LOCAL_INTEGRATION") != "1",
        reason="set RUN_LOCAL_INTEGRATION=1 for local waveform generation",
    ),
    pytest.mark.skipif(
        not (HAS_INT8 or HAS_ORIGINAL),
        reason="VibeVoice checkpoint is not present",
    ),
]


def test_short_generation() -> None:
    loaded = load_vibevoice_model(MODEL_DIR, strict=False)
    tokenizer = VibeVoiceTokenizer.from_path(MODEL_DIR)
    config = VibeVoiceGenerationConfig(max_new_tokens=20, do_sample=False)

    result = synthesize_vibevoice(loaded.model, tokenizer, "Hello.", config=config)
    mx.eval(result.waveform)

    assert isinstance(result, VibeVoiceSynthesisOutput)
    assert result.sample_rate == 24_000
    assert result.generated_tokens > 0
    assert result.waveform.shape[0] > 0


def test_voice_cloning() -> None:
    reference_path = Path("outputs/source/hank_hill_ref.wav")
    if not reference_path.is_file():
        pytest.skip("reference audio is not present")

    loaded = load_vibevoice_model(MODEL_DIR, strict=False)
    tokenizer = VibeVoiceTokenizer.from_path(MODEL_DIR)
    config = VibeVoiceGenerationConfig(max_new_tokens=20, do_sample=False)
    reference, _ = load_audio(str(reference_path), sample_rate=24_000)

    result = synthesize_vibevoice(
        loaded.model,
        tokenizer,
        "Hello.",
        reference_audio=reference.reshape(1, 1, -1),
        config=config,
    )
    mx.eval(result.waveform)

    assert result.waveform.shape[0] > 0
