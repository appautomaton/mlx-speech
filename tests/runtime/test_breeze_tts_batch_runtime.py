"""Real Breeze codec batches preserve each causal audio prefix."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.generation.breeze_tts_batch import _decode_audio
from mlx_speech.tts import load


@pytest.fixture(scope="module")
def model():
    path = Path("models/breezeblue/breeze_tts_2/mlx-int8")
    if not (path / "model.safetensors").exists():
        pytest.skip("Local Breeze int8 checkpoint is missing")
    return load(str(path))


@pytest.mark.runtime
def test_batched_codec_trims_padding_without_changing_real_audio(model):
    codes = [mx.arange(length * 16).reshape(length, 16) % 2048 for length in (3, 5)]
    together = _decode_audio(model._codec, codes)
    for row, result in zip(codes, together, strict=True):
        single = model._codec.decode(row.T[None])[0, 0]
        assert result.samples.shape == single.shape
        assert mx.allclose(result.samples, single, atol=1e-4).item()
