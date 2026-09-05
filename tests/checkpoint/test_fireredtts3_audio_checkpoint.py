from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fireredtts3.redae import RedAE
from mlx_speech.models.fireredtts3.speaker import FireRedSpeakerEncoder


ARTIFACT = Path("models/firered/firered_tts3/mlx-bf16")

pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "redae.safetensors").is_file(),
    reason="local FireRedTTS3 BF16 artifact is unavailable",
)


def test_fireredtts3_audio_components_strict_load_and_execute() -> None:
    speaker = FireRedSpeakerEncoder.from_dir(ARTIFACT)
    time = np.arange(16_000, dtype=np.float32) / 16_000
    waveform = 0.05 * np.sin(2 * np.pi * 220.0 * time)
    embedding = speaker(waveform, sample_rate=16_000)
    mx.eval(embedding)
    assert embedding.shape == (1, 512)
    assert bool(mx.all(mx.isfinite(embedding)).item())
    del speaker
    gc.collect()
    mx.clear_cache()

    redae = RedAE.from_dir(ARTIFACT)
    audio = mx.array(waveform[:3_840][None], dtype=mx.float32)
    latents = redae.encode(audio)
    decoded = redae.decode(latents)
    mx.eval(latents, decoded)
    assert latents.shape == (1, 4, 64)
    assert decoded.shape == (1, 3_840)
    assert bool(mx.all(mx.isfinite(latents)).item())
    assert bool(mx.all(mx.isfinite(decoded)).item())
