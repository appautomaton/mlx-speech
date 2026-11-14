from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.feature_extraction import (
    Qwen3ASRFeatureExtractor,
    _get_feat_extract_output_lengths,
    _get_mel_filters,
    _hann_window,
    _spectrogram,
)


def _deterministic_waveform(samples: int = 1600) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_qwen3_asr_feature_extractor_reads_preprocessor_config(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        """{
            "sampling_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "feature_size": 128,
            "chunk_length": 30,
            "padding_value": 0.0,
            "dither": 0.0
        }""",
        encoding="utf-8",
    )

    extractor = Qwen3ASRFeatureExtractor.from_dir(tmp_path)

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


def _spectrogram_loop(waveform, *, window, frame_length, hop_length, mel_filters, dither):
    """Pre-vectorization frontend, including its per-frame dither draws."""
    waveform = np.pad(
        waveform.astype(np.float64), [(frame_length // 2, frame_length // 2)], mode="reflect",
    )
    window = window.astype(np.float64)
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))
    spectrum = np.empty((num_frames, frame_length // 2 + 1), dtype=np.float32)
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        frame = waveform[start : start + frame_length].copy()
        if dither != 0.0:
            frame += dither * np.random.randn(frame_length)
        frame *= window
        fft = np.fft.rfft(frame)
        spectrum[frame_idx] = (np.abs(fft, dtype=np.float64) ** 2).astype(np.float32)
    mel_spec = np.maximum(1e-10, mel_filters.T @ spectrum.T)
    return np.log10(mel_spec).astype(np.float32)


@pytest.mark.parametrize("samples", [160, 1600, 1733, 128000])
@pytest.mark.parametrize("dither", [0.0, 0.001])
def test_qwen3_asr_vectorized_spectrogram_matches_loop(samples, dither):
    waveform = np.random.default_rng(74).normal(0, 0.1, samples).astype(np.float32)
    kwargs = dict(
        window=_hann_window(400), frame_length=400, hop_length=160,
        mel_filters=_get_mel_filters(sample_rate=16000, n_fft=400, n_mels=128), dither=dither,
    )
    state = np.random.get_state()
    try:
        np.random.seed(75)
        expected = _spectrogram_loop(waveform, **kwargs)
        np.random.seed(75)
        actual = _spectrogram(waveform, **kwargs)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(actual, expected)
