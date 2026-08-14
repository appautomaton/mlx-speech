"""Real-checkpoint runtime coverage for the VibeVoice acoustic tokenizer."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model


MODEL_DIR = Path("models/vibevoice/mlx-int8")

pytestmark = [
    pytest.mark.runtime,
    pytest.mark.skipif(
        not (MODEL_DIR / "config.json").is_file(),
        reason="VibeVoice checkpoint is not present",
    ),
]


def test_encoder_output_shape() -> None:
    loaded = load_vibevoice_model(MODEL_DIR, strict=False)
    encoder = loaded.model.model.acoustic_tokenizer.encoder
    waveform = mx.random.normal((1, 1, 24_000))
    output = encoder(waveform)
    mx.eval(output)

    assert output.shape[0] == 1
    assert output.shape[1] == 64
    assert output.shape[2] >= 7


def test_encode_decode_roundtrip_shape() -> None:
    loaded = load_vibevoice_model(MODEL_DIR, strict=False)
    tokenizer = loaded.model.model.acoustic_tokenizer
    waveform = mx.random.normal((1, 1, 24_000))
    latent = tokenizer.encode(waveform)
    reconstructed = tokenizer.decode(latent)
    mx.eval(latent, reconstructed)

    assert reconstructed.shape[0] == 1
    assert reconstructed.shape[1] == 1
    assert abs(reconstructed.shape[2] - 24_000) < 3_200
