from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fireredtts3.core import FireRedTTS3Core
from mlx_speech.models.fireredtts3.tokenizer import FireRedTTS3Tokenizer


ARTIFACT = Path("models/firered/firered_tts3/mlx-bf16")

pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "core.safetensors").is_file(),
    reason="local FireRedTTS3 BF16 artifact is unavailable",
)


def test_real_tokenizer_matches_official_base_ids() -> None:
    tokenizer = FireRedTTS3Tokenizer.from_dir(ARTIFACT)
    assert tokenizer.encode(
        language="Chinese",
        reference_text="参考",
        text="你好",
    ) == [151876, 151676, 101275, 108386, 151677]


def test_real_core_strict_loads_and_generates_one_finite_patch() -> None:
    core = FireRedTTS3Core.from_dir(ARTIFACT)
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
    assert result.generated_patches == 1
    assert result.latents.shape == (1, 8, 64)
    assert bool(mx.all(mx.isfinite(result.latents)).item())
    del core
    gc.collect()
    mx.clear_cache()
