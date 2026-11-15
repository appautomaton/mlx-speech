"""The upstream FireRedTTS3 release, read in RAM at upstream precision."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.fireredtts3.checkpoint import upstream_dtype
from mlx_speech.models.fireredtts3.core import FireRedTTS3Core
from mlx_speech.models.fireredtts3.redae import RedAE
from mlx_speech.models.fireredtts3.speaker import FireRedSpeakerEncoder

UPSTREAM = Path("models/firered/firered_tts3/original")
GiB = 2**30

pytestmark = pytest.mark.skipif(
    not (UPSTREAM / "fireredtts3_base" / "model.safetensors").is_file(),
    reason="local FireRedTTS3 upstream release is unavailable",
)


def release(*models) -> None:
    del models
    gc.collect()
    mx.clear_cache()


def assert_upstream_precision(component: str, module) -> None:
    for name, value in tree_flatten(module.parameters()):
        assert value.dtype == upstream_dtype(component, name), name


def tone(sample_rate: int, samples: int) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / sample_rate
    return 0.05 * np.sin(2 * np.pi * 220.0 * time) * (1 + np.sin(2 * np.pi * 3 * time))


def test_core_loads_without_holding_the_fp32_file_and_generates() -> None:
    release()
    mx.reset_peak_memory()
    core = FireRedTTS3Core.from_upstream(UPSTREAM)
    mx.eval(core.parameters())
    # BF16 backbone linears plus FP32 rest is about 5.3 GiB; the FP32 file is 7.9 GB.
    assert mx.get_peak_memory() < 6 * GiB
    assert_upstream_precision("core", core)

    result = core.generate(
        speaker_embedding=mx.zeros((1, 512), dtype=mx.float32),
        text_tokens=mx.array([[151876, 151676, 151677]], dtype=mx.int32),
        prompt_latents=mx.zeros((1, 4, 64), dtype=mx.float32),
        flow_steps=2,
        guidance_scale=0.0,
        stop_threshold=2.0,
        min_generated_patches=0,
        max_generated_patches=2,
        seed=23,
    )
    mx.eval(result.latents)
    assert result.latents.shape == (1, 12, 64)
    assert bool(mx.all(mx.isfinite(result.latents)).item())
    assert all(0.0 <= score <= 1.0 for score in result.stop_scores)
    release(core)


def test_audio_components_load_at_upstream_precision_and_run() -> None:
    speaker = FireRedSpeakerEncoder.from_upstream(UPSTREAM)
    assert_upstream_precision("speaker", speaker.model)
    embedding = speaker(tone(16_000, 16_000), sample_rate=16_000)
    mx.eval(embedding)
    assert embedding.shape == (1, 512)
    assert bool(mx.all(mx.isfinite(embedding)).item())
    release(speaker)

    redae = RedAE.from_upstream(UPSTREAM)
    assert_upstream_precision("redae", redae)
    assert not any("embed_tokens" in name for name, _ in tree_flatten(redae.parameters()))
    audio = tone(24_000, 23_040)
    decoded = np.array(redae.decode(redae.encode(mx.array(audio[None]))))[0]
    assert decoded.shape == audio.shape
    # 22.3 dB as loaded; swapping two decoder tensors already drops it to 16.9 dB.
    snr = 10 * np.log10((audio**2).sum() / ((audio - decoded) ** 2).sum())
    assert snr > 20, snr
    release(redae)
