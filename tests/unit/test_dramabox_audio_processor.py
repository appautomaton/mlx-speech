from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dramabox.audio_vae.audio_processor import AudioProcessor


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dramabox"
    / "audio_processor_mel_fixture.npz"
)


def test_waveform_to_mel_shape_for_stereo_input():
    processor = AudioProcessor()
    waveform = mx.zeros((1, 2, 1600), dtype=mx.float32)

    mel = processor.waveform_to_mel(waveform, sample_rate=16_000)

    assert mel.shape == (1, 2, 11, 64)
    assert mx.all(mx.isfinite(mel)).item()


def test_waveform_to_mel_matches_torchaudio_fixture():
    fixture = np.load(FIXTURE)
    processor = AudioProcessor()

    actual = processor.waveform_to_mel(
        mx.array(fixture["waveform"], dtype=mx.float32),
        sample_rate=int(fixture["sample_rate"]),
    )
    mx.eval(actual)

    actual_np = np.asarray(actual)
    expected = fixture["log_mel"].astype(np.float32)
    assert actual_np.shape == expected.shape
    assert float(np.max(np.abs(actual_np - expected))) <= 1e-2


def test_waveform_to_mel_rejects_non_batched_waveform():
    processor = AudioProcessor()

    with pytest.raises(ValueError, match="expects"):
        processor.waveform_to_mel(mx.zeros((2, 1600), dtype=mx.float32), sample_rate=16_000)
