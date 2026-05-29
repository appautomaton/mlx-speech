from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mlx_speech.models.granite_speech_asr.feature_extraction import (
    GraniteSpeechFeatureExtractor,
    _htk_mel_filters,
)


def test_granite_feature_extractor_uses_reference_settings():
    extractor = GraniteSpeechFeatureExtractor()

    assert extractor.sample_rate == 16000
    assert extractor.n_fft == 512
    assert extractor.win_length == 400
    assert extractor.hop_length == 160
    assert extractor.n_mels == 80
    assert extractor.window_size == 15
    assert extractor.downsample_rate == 5
    assert _htk_mel_filters(16000, 512, 80).shape == (80, 257)


def test_granite_feature_extractor_pair_stacks_to_encoder_input():
    extractor = GraniteSpeechFeatureExtractor()
    waveform = np.zeros(16000, dtype=np.float32)

    features, audio_tokens = extractor(waveform)

    assert features.shape == (1, 50, 160)
    assert features.dtype == np.float32
    assert audio_tokens == 12


def test_granite_feature_extractor_trims_odd_mel_frames():
    extractor = GraniteSpeechFeatureExtractor()
    waveform = np.zeros(320, dtype=np.float32)

    features, audio_tokens = extractor(waveform)

    assert extractor.preflight_shape(len(waveform)).mel_frames == 3
    assert features.shape == (1, 1, 160)
    assert audio_tokens == 3


def test_granite_audio_token_count_matches_window_formula():
    extractor = GraniteSpeechFeatureExtractor()
    shape = extractor.preflight_shape(16000)

    assert shape.mel_frames == 101
    assert shape.encoder_frames == 50
    assert shape.audio_tokens == math.ceil(50 / 15) * (15 // 5)


def test_granite_feature_extractor_from_dir_reads_preprocessor(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "granite_speech",
                "audio_token_index": 100352,
                "downsample_rate": 5,
                "window_size": 15,
                "encoder_config": {"input_dim": 160},
                "projector_config": {},
                "text_config": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "melspec_kwargs": {
                    "hop_length": 80,
                    "n_fft": 256,
                    "n_mels": 40,
                    "sample_rate": 8000,
                    "win_length": 200,
                },
                "projector_downsample_rate": 4,
                "projector_window_size": 12,
            }
        ),
        encoding="utf-8",
    )

    extractor = GraniteSpeechFeatureExtractor.from_dir(tmp_path)

    assert extractor.sample_rate == 8000
    assert extractor.n_fft == 256
    assert extractor.win_length == 200
    assert extractor.hop_length == 80
    assert extractor.n_mels == 40
    assert extractor.window_size == 12
    assert extractor.downsample_rate == 4


def test_granite_feature_extractor_rejects_non_mono_waveform():
    extractor = GraniteSpeechFeatureExtractor()

    with pytest.raises(ValueError, match="1D mono"):
        extractor(np.zeros((1, 16000), dtype=np.float32))
