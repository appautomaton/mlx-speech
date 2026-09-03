from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fireredtts3.core import (
    CoreGenerationResult,
    FireRedTTS3Core,
)


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "models/firered/firered_tts3/mlx-bf16"
MAX_ACTIVE_MEMORY_SPREAD = 64 * 1024 * 1024

pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "core.safetensors").is_file(),
    reason="local FireRedTTS3 BF16 artifact is unavailable",
)


def _generate_one_patch(core: FireRedTTS3Core) -> CoreGenerationResult:
    result = core.generate(
        speaker_embedding=mx.zeros((1, 512), dtype=mx.float32),
        text_tokens=mx.array([[151876, 151676, 151677]], dtype=mx.int32),
        prompt_latents=mx.zeros((1, 4, 64), dtype=mx.float32),
        flow_steps=1,
        guidance_scale=0.0,
        stop_threshold=2.0,
        min_generated_patches=0,
        max_generated_patches=1,
        seed=23,
    )
    mx.eval(result.latents)
    return result


def test_repeated_core_requests_release_request_memory() -> None:
    core = FireRedTTS3Core.from_dir(ARTIFACT)
    assert core._compiled_dit is not None

    warmup = _generate_one_patch(core)
    assert warmup.generated_patches == 1
    del warmup
    gc.collect()
    mx.clear_cache()

    active_memory: list[int] = []
    for _ in range(3):
        result = _generate_one_patch(core)
        assert result.generated_patches == 1
        assert result.cache_length == result.prompt_length
        del result
        gc.collect()
        mx.clear_cache()
        active_memory.append(int(mx.get_active_memory()))

    assert max(active_memory) - min(active_memory) <= MAX_ACTIVE_MEMORY_SPREAD

    del core
    gc.collect()
    mx.clear_cache()
