from __future__ import annotations

import gc
import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech import tts
from mlx_speech.tts import TTSOutput


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("variant", ["soar", "mf"])
def test_unified_tts_api_generates_base_waveform(variant: str) -> None:
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("set RUN_LOCAL_INTEGRATION=1 for local checkpoint generation")
    model_dir = ROOT / "models/dots_tts" / variant / "mlx-base"
    if not (model_dir / "core.safetensors").is_file():
        pytest.skip(f"local dots.tts {variant} base checkpoint is unavailable")
    model = tts.load(str(model_dir))
    result = model.generate(
        "Unified API waveform check.",
        max_new_tokens=1,
        solver_steps=1,
        seed=23,
        eos_threshold=1.0,
    )
    assert isinstance(result, TTSOutput)
    assert result.sample_rate == 48_000
    assert result.waveform.ndim == 1
    assert int(result.waveform.size) > 0
    assert bool(mx.all(mx.isfinite(result.waveform)).item())
    assert bool(mx.any(mx.abs(result.waveform) > 0).item())
    del model, result
    gc.collect()
    mx.clear_cache()
