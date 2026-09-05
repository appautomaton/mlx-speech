from __future__ import annotations

import gc
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech import asr, tts
from mlx_speech.audio import load_audio, resample_audio, write_wav


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "models/firered/firered_tts3/mlx-bf16"
ASR_ARTIFACT = ROOT / "models/qwen3_asr_1_7b/mlx-int8"
REFERENCE = Path("/tmp/fireredtts3-mlx-port/reference.wav")
OUTPUT = Path("/tmp/fireredtts3-mlx-port/generated-mlx.wav")
TARGET_TEXT = "你好，很高兴认识你。"
REFERENCE_TEXT = "For Timothy was a spoiled cat, and he allowed no one."


def test_fireredtts3_public_api_generates_and_transcribes_waveform() -> None:
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("set RUN_LOCAL_INTEGRATION=1 for local checkpoint generation")
    if not (ARTIFACT / "core.safetensors").is_file():
        pytest.skip("local FireRedTTS3 BF16 artifact is unavailable")
    if not (ASR_ARTIFACT / "model.safetensors").is_file():
        pytest.skip("local Qwen3-ASR artifact is unavailable")
    if not REFERENCE.is_file():
        pytest.skip(f"reference audio is unavailable: {REFERENCE}")

    model = tts.load(str(ARTIFACT))
    generation_args = {
        "reference_audio": REFERENCE,
        "reference_text": REFERENCE_TEXT,
        "language": "Chinese",
        "seed": 1234,
        "guidance_scale": 2.0,
        "flow_steps": 10,
        "stop_threshold": 0.5,
        "max_audio_patches": 40,
    }
    result = model.generate(
        TARGET_TEXT,
        **generation_args,
    )
    repeated = model.generate(
        TARGET_TEXT,
        **generation_args,
    )
    mx.eval(result.waveform, repeated.waveform)
    np.testing.assert_array_equal(
        np.asarray(result.waveform),
        np.asarray(repeated.waveform),
    )
    assert result.sample_rate == 24_000
    assert result.waveform.ndim == 1
    assert int(result.waveform.size) > 4_800
    assert bool(mx.all(mx.isfinite(result.waveform)).item())
    assert float(mx.max(mx.abs(result.waveform)).item()) > 1e-4
    write_wav(OUTPUT, result.waveform, sample_rate=result.sample_rate)
    reference_waveform, reference_rate = load_audio(REFERENCE, mono=True)
    reference_embedding = model.speaker(
        reference_waveform,
        sample_rate=reference_rate,
    ).astype(mx.float32)
    output_embedding = model.speaker(
        result.waveform,
        sample_rate=result.sample_rate,
    ).astype(mx.float32)
    speaker_cosine = mx.sum(reference_embedding * output_embedding) / (
        mx.sqrt(mx.sum(reference_embedding * reference_embedding))
        * mx.sqrt(mx.sum(output_embedding * output_embedding))
    )
    speaker_cosine_value = float(speaker_cosine.item())
    assert speaker_cosine_value >= 0.70, speaker_cosine_value
    waveform = np.asarray(result.waveform, dtype=np.float32)

    del model, repeated, result
    gc.collect()
    mx.clear_cache()

    transcriber = asr.load(str(ASR_ARTIFACT))
    asr_waveform = resample_audio(
        mx.array(waveform),
        orig_sample_rate=24_000,
        target_sample_rate=16_000,
    )
    transcription = transcriber.generate(
        asr_waveform,
        sample_rate=16_000,
        language="Chinese",
    )
    assert transcription.text == TARGET_TEXT
