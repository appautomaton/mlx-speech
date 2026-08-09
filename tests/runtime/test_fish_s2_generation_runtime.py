from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.generation.fish_s2_pro import generate_fish_s2_pro
from mlx_speech.models.fish_s2_pro.checkpoint import load_fish_s2_pro_checkpoint


def test_local_checkpoint_loads_real_repacked_shards():
    model_dir = Path("models/fish_s2_pro/mlx-int8")
    if not model_dir.exists():
        pytest.skip("No local Fish S2 checkpoint")
    ckpt = load_fish_s2_pro_checkpoint(model_dir)
    assert "embeddings.weight" in ckpt.state_dict


@pytest.mark.runtime
def test_runtime_generates_waveform_with_local_assets():
    model_dir = Path("models/fish_s2_pro/mlx-int8")
    if not (model_dir / "codec-mlx").is_dir():
        pytest.skip("Fish local assets not ready")

    out = generate_fish_s2_pro(
        "Hello from Fish S2.",
        model_dir=model_dir,
        max_new_tokens=8,
    )

    assert out.waveform.ndim == 1
    assert out.waveform.shape[0] > 0
    assert mx.all(mx.isfinite(out.waveform)).item()
    assert mx.max(mx.abs(out.waveform)).item() > 0
    assert out.sample_rate == 44100
    assert out.generated_tokens > 0
