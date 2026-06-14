from pathlib import Path

import mlx.core as mx
import pytest

import mlx_speech

DRAMABOX_DIR = Path("models/dramabox/mlx-bf16")
GEMMA_DIR = Path("models/gemma_3_12b_it_backbone/mlx-4bit")


@pytest.mark.runtime
def test_dramabox_alias_loads_and_generates_stereo():
    """tts.load dispatches DramaBox and produces a 48 kHz stereo waveform.

    Uses local paths (gemma override) so the test never hits the Hub.
    """
    if not DRAMABOX_DIR.exists() or not GEMMA_DIR.exists():
        pytest.skip("DramaBox local assets not ready")

    model = mlx_speech.tts.load(str(DRAMABOX_DIR), gemma_path_or_repo=str(GEMMA_DIR))
    assert type(model).__name__ == "DramaBoxAdapter"

    out = model.generate("A quick test of the DramaBox alias.", duration_seconds=2.0)

    assert out.sample_rate == 48000
    assert out.waveform.ndim == 2
    assert out.waveform.shape[0] == 2  # stereo
    assert out.waveform.shape[1] > 0
    assert mx.all(mx.isfinite(out.waveform)).item()
    assert mx.max(mx.abs(out.waveform)).item() > 0
