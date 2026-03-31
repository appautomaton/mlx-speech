from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_feature_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mlx_speech"
        / "models"
        / "cohere_asr"
        / "feature_extraction.py"
    )
    spec = importlib.util.spec_from_file_location("cohere_asr_feature_extraction_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_features_length_matches_upstream_formula() -> None:
    module = _load_feature_module()
    extractor = module.CohereAsrFeatureExtractor(dither=0.0, preemphasis=0.0)

    assert extractor._features_length(0) == 0
    assert extractor._features_length(160) == 1
    assert extractor._features_length(48000) == 300
    assert extractor._features_length(64000) == 400

    features, attention_mask = extractor(np.zeros(48000, dtype=np.float32))
    assert len(features) == 301
    assert int(attention_mask.sum()) == 300


def test_normalization_uses_sample_variance(monkeypatch) -> None:
    module = _load_feature_module()
    extractor = module.CohereAsrFeatureExtractor(dither=0.0, preemphasis=0.0)

    fake_features = np.array(
        [
            [1.0, 4.0],
            [3.0, 8.0],
            [5.0, 12.0],
        ],
        dtype=np.float32,
    )

    def fake_log_mel(*args, **kwargs):
        return fake_features.copy()

    monkeypatch.setattr(module, "_log_mel_spectrogram", fake_log_mel)

    features, attention_mask = extractor(np.zeros(320, dtype=np.float32))

    expected = np.array(
        [
            [-0.70710677, -0.70710677],
            [0.70710677, 0.70710677],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(attention_mask, np.array([True, True, False]))


def test_from_dir_reads_preprocessor_and_chunk_settings(tmp_path: Path) -> None:
    module = _load_feature_module()

    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "sampling_rate": 22050,
                "n_fft": 1024,
                "n_window_stride": 256,
                "n_window_size": 800,
                "feature_size": 80,
                "preemphasis": 0.95,
                "dither": 2e-5,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "sample_rate": 22050,
                "max_audio_clip_s": 12.5,
                "overlap_chunk_second": 1.5,
            }
        ),
        encoding="utf-8",
    )

    extractor = module.CohereAsrFeatureExtractor.from_dir(tmp_path)

    assert extractor.sr == 22050
    assert extractor.n_fft == 1024
    assert extractor.hop_length == 256
    assert extractor.win_length == 800
    assert extractor.n_mels == 80
    assert extractor.preemphasis == 0.95
    assert extractor.dither == 2e-5
    assert extractor.max_audio_clip_s == 12.5
    assert extractor.overlap_chunk_s == 1.5
