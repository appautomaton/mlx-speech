"""The public Breeze batch API reaches waveform output with real weights."""

import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.tts import load


@pytest.fixture(scope="module")
def model():
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("Set RUN_LOCAL_INTEGRATION=1 for local waveform generation")
    path = Path("models/breezeblue/breeze_tts_2/mlx-int8")
    if not (path / "model.safetensors").exists():
        pytest.skip("Local Breeze int8 checkpoint is missing")
    return load(str(path))


@pytest.mark.integration
@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_public_batch_api_returns_real_waveforms(model, scale):
    with pytest.warns(RuntimeWarning, match="frame budget"):
        output = model.generate_batch(
            ["你好。", "今天的天气很好。"],
            guidance_scale=scale,
            instruction="自然、清晰地朗读。" if scale != 1 else None,
            max_new_tokens=4,
            seeds=[11, 12],
            temperature=0.9,
        )
    assert len(output) == 2
    for item in output:
        assert item.sample_rate == 24_000
        assert item.waveform.shape == (4 * 1920,)
        assert mx.all(mx.isfinite(item.waveform)).item()
        assert mx.max(mx.abs(item.waveform)).item() > 1e-5
