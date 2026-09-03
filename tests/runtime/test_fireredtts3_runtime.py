from __future__ import annotations

import gc
import math
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.tts._adapter import TTSOutput
from mlx_speech.tts._adapters.fireredtts3 import FireRedTTS3Adapter


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "models/firered/firered_tts3/mlx-bf16"
MAX_ACTIVE_MEMORY_SPREAD = 64 * 1024 * 1024

pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "core.safetensors").is_file(),
    reason="local FireRedTTS3 BF16 artifact is unavailable",
)


def _generate_one_patch(
    model: FireRedTTS3Adapter,
    reference: mx.array,
) -> TTSOutput:
    result = model.generate(
        "Runtime memory check.",
        reference_audio=reference,
        reference_text="Reference tone.",
        reference_sample_rate=24_000,
        flow_steps=1,
        guidance_scale=0.0,
        stop_threshold=1.0,
        max_audio_patches=1,
        seed=23,
    )
    mx.eval(result.waveform)
    return result


def test_repeated_public_requests_release_request_memory() -> None:
    model = FireRedTTS3Adapter.from_dir(ARTIFACT)
    assert model.core._compiled_dit is not None
    time = mx.arange(96_000, dtype=mx.float32) / 24_000.0
    reference = 0.1 * mx.sin(2.0 * math.pi * 220.0 * time)

    warmup = _generate_one_patch(model, reference)
    assert warmup.sample_rate == 24_000
    del warmup
    gc.collect()
    mx.clear_cache()

    active_memory: list[int] = []
    for _ in range(3):
        result = _generate_one_patch(model, reference)
        assert result.sample_rate == 24_000
        assert bool(mx.all(mx.isfinite(result.waveform)).item())
        del result
        gc.collect()
        mx.clear_cache()
        active_memory.append(int(mx.get_active_memory()))

    assert max(active_memory) - min(active_memory) <= MAX_ACTIVE_MEMORY_SPREAD

    del model, reference, time
    gc.collect()
    mx.clear_cache()
