from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.nemotron_asr.feature_extraction import (
    NemotronFeatureExtractor,
    _slaney_mel_filters,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "nemotron" / "features.npz"


def test_feature_config_matches_checkpoint() -> None:
    extractor = NemotronFeatureExtractor()

    assert extractor.sample_rate == 16_000
    assert extractor.n_fft == 512
    assert extractor.win_length == 400
    assert extractor.hop_length == 160
    assert extractor.n_mels == 128
    assert extractor.preemphasis == 0.97
    assert extractor.dither == 1e-5
    assert extractor.normalize == "NA"
    assert extractor.log_zero_guard_value == 2**-24


def test_slaney_filters_have_reference_shape_and_area_normalization() -> None:
    filters = _slaney_mel_filters(16_000, 512, 128)

    assert filters.shape == (128, 257)
    assert filters.dtype == np.float32
    assert np.all(filters >= 0.0)
    assert np.all(np.count_nonzero(filters, axis=1) > 0)


def test_features_match_captured_nemo_reference() -> None:
    with np.load(_FIXTURE) as fixture:
        waveform = fixture["waveform"]
        expected = fixture["features"]
        expected_length = fixture["length"]

    features, length = NemotronFeatureExtractor()(waveform)
    mx.eval(features, length)

    np.testing.assert_allclose(np.asarray(features), expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_array_equal(np.asarray(length), expected_length)


def test_normalize_na_is_a_noop_and_dither_is_disabled_at_inference() -> None:
    waveform = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    extractor = NemotronFeatureExtractor()

    first, first_length = extractor(waveform)
    second, second_length = extractor(waveform)
    mx.eval(first, second, first_length, second_length)

    assert mx.array_equal(first, second).item()
    assert mx.array_equal(first_length, second_length).item()
    assert not np.allclose(np.asarray(first)[0, : int(first_length[0])].mean(axis=0), 0.0)


def test_final_center_padded_frame_is_masked_like_nemo() -> None:
    features, length = NemotronFeatureExtractor()(np.ones(320, dtype=np.float32))
    mx.eval(features, length)

    assert features.shape == (1, 3, 128)
    assert int(length[0]) == 2
    assert mx.all(features[0, -1] == 0.0).item()


@pytest.mark.parametrize("shape", [(1, 16), (2, 8)])
def test_rejects_non_mono_waveform(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="1D mono"):
        NemotronFeatureExtractor()(np.zeros(shape, dtype=np.float32))


def test_rejects_empty_waveform() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NemotronFeatureExtractor()(np.zeros((0,), dtype=np.float32))


def test_rejects_feature_normalization() -> None:
    with pytest.raises(NotImplementedError, match="normalize='NA'"):
        NemotronFeatureExtractor(normalize="per_feature")
