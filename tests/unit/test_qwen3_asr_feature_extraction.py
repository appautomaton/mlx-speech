from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.feature_extraction import (
    Qwen3ASRFeatureExtractor,
    _get_feat_extract_output_lengths,
)


QWEN_DIR = Path("models/qwen3_asr_1_7b/original")


def _deterministic_waveform(samples: int = 1600) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_qwen3_asr_feature_extractor_reads_preprocessor_config():
    if not (QWEN_DIR / "preprocessor_config.json").exists():
        pytest.skip("Qwen3-ASR preprocessor assets not present")

    extractor = Qwen3ASRFeatureExtractor.from_dir(QWEN_DIR)

    assert extractor.sample_rate == 16000
    assert extractor.n_fft == 400
    assert extractor.hop_length == 160
    assert extractor.n_mels == 128
    assert extractor.n_samples == 480000
    assert extractor.nb_max_frames == 3000


def test_qwen3_asr_feature_extractor_dynamic_padding_and_masks():
    extractor = Qwen3ASRFeatureExtractor()
    short = _deterministic_waveform(1600)
    long = _deterministic_waveform(3200)

    batch = extractor([short, long], sample_rate=16000)

    assert batch.input_features.shape == (2, 128, 20)
    assert batch.feature_attention_mask.shape == (2, 20)
    assert batch.feature_attention_mask[0].sum() == 10
    assert batch.feature_attention_mask[1].sum() == 20
    assert batch.audio_lengths.tolist() == [2, 3]


def test_qwen3_asr_feature_extractor_reference_vector_snapshot():
    extractor = Qwen3ASRFeatureExtractor()

    batch = extractor(_deterministic_waveform(), sample_rate=16000)

    expected = np.array(
        [
            [0.55803454, 0.04602641, -0.86413085],
            [0.65559900, 0.14359087, -0.86413085],
            [0.63692010, 0.12601936, -0.86413085],
            [0.60246533, 0.09156460, -0.86413085],
            [0.69303800, 0.17686617, -0.86413085],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(batch.input_features[0, :5, :3], expected, atol=1e-5)


def test_qwen3_asr_feature_extractor_rejects_empty_and_non_mono():
    extractor = Qwen3ASRFeatureExtractor()

    with pytest.raises(ValueError, match="non-empty"):
        extractor(np.array([], dtype=np.float32), sample_rate=16000)

    with pytest.raises(ValueError, match="1D mono"):
        extractor(np.zeros((1, 1600), dtype=np.float32), sample_rate=16000)


def test_qwen3_asr_feature_extractor_rejects_wrong_array_sample_rate():
    extractor = Qwen3ASRFeatureExtractor()

    with pytest.raises(ValueError, match="Resample before"):
        extractor(_deterministic_waveform(), sample_rate=8000)


def test_qwen3_asr_feature_extractor_accepts_path_audio(monkeypatch, tmp_path):
    extractor = Qwen3ASRFeatureExtractor()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not used")

    def fake_load_audio(path, *, sample_rate, mono):
        assert path == audio_path
        assert sample_rate == 16000
        assert mono is True
        return _deterministic_waveform(), 16000

    monkeypatch.setitem(
        sys.modules,
        "mlx_speech.audio",
        SimpleNamespace(load_audio=fake_load_audio),
    )

    batch = extractor(audio_path)

    assert batch.input_features.shape == (1, 128, 10)
    assert batch.feature_attention_mask.sum() == 10


def test_qwen3_asr_feature_output_length_formula_matches_reference_cases():
    lengths = np.array([0, 1, 10, 100, 101, 1000], dtype=np.int64)

    assert _get_feat_extract_output_lengths(lengths).tolist() == [0, 1, 2, 13, 14, 130]
    assert _get_feat_extract_output_lengths(100) == 13
