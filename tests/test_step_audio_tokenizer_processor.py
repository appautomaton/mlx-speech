"""Tests for the Step-Audio tokenizer processor layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioTokenizerAssets,
    StepAudioTokenizerConfig,
    StepAudioTokenizerProcessor,
)


def _make_assets(
    *,
    codebook: np.ndarray | None = None,
    config: StepAudioTokenizerConfig | None = None,
) -> StepAudioTokenizerAssets:
    return StepAudioTokenizerAssets(
        model_dir=Path("/tmp/step-audio-tokenizer"),
        config=config or StepAudioTokenizerConfig(),
        linguistic_tokenizer_path=Path("/tmp/linguistic_tokenizer.npy"),
        semantic_tokenizer_path=Path("/tmp/speech_tokenizer_v1.onnx"),
        funasr_model_dir=Path("/tmp/funasr"),
        funasr_config_path=Path("/tmp/funasr/config.yaml"),
        funasr_checkpoint_path=Path("/tmp/funasr/model.pt"),
        linguistic_codebook=np.asarray(
            codebook
            if codebook is not None
            else np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [5.0, 5.0],
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
    )


def test_preprocess_wav_resamples_and_energy_normalizes() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    audio = np.ones((1, 3200), dtype=np.float32) * 0.25

    processed = processor.preprocess_wav(
        audio,
        sample_rate=32000,
        enable_trim=False,
        energy_norm=True,
    )

    assert processed.dtype == np.float32
    assert processed.shape == (1600,)
    assert np.isclose(np.max(np.abs(processed)), 0.999, atol=1e-3)


def test_preprocess_wav_trim_keeps_context_and_reduces_leading_silence() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    audio = np.concatenate(
        [
            np.zeros(8000, dtype=np.float32),
            np.ones(3200, dtype=np.float32) * 0.4,
            np.zeros(1600, dtype=np.float32),
        ]
    )

    processed = processor.preprocess_wav(
        audio,
        sample_rate=16000,
        enable_trim=True,
        energy_norm=False,
    )

    assert processed.shape[0] < audio.shape[0]
    assert np.allclose(processed[:100], 0.0)
    assert float(np.max(processed)) > 0.35


def test_cluster_linguistic_features_assigns_nearest_codebook_rows() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    features = np.array(
        [
            [0.1, 0.1],
            [0.9, 1.1],
            [4.8, 5.2],
        ],
        dtype=np.float32,
    )

    labels = processor.cluster_linguistic_features(features)
    assert labels == [0, 1, 2]


def test_dump_label_accepts_sample_batches() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    sample_a = np.array([[[0.1, 0.1], [1.2, 0.8]]], dtype=np.float32)
    sample_b = np.array([[[4.9, 5.1]]], dtype=np.float32)

    assert processor.dump_label([sample_a, sample_b]) == [[0, 1], [2]]


def test_split_vq06_audio_chunks_drops_short_tail() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    max_chunk = int(processor.config.vq06_max_chunk_seconds * processor.config.vq06_sample_rate)
    audio = np.zeros((max_chunk * 2 + 479,), dtype=np.float32)

    chunks = processor.split_vq06_audio(audio)

    assert len(chunks) == 2
    assert all(chunk.shape == (max_chunk,) for chunk in chunks)


def test_compute_vq06_log_mel_spectrogram_shape_matches_whisper_contract() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    samples = np.sin(2.0 * np.pi * 440.0 * np.arange(16000, dtype=np.float32) / 16000.0)

    features = processor.compute_vq06_log_mel_spectrogram(samples)

    assert features.shape == (128, 100)
    assert np.isfinite(features).all()
    assert float(np.max(features)) > 0.5
    assert float(np.min(features)) < 0.0


def test_prepare_vq06_chunks_builds_expected_metadata() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    samples = np.sin(2.0 * np.pi * 220.0 * np.arange(16000, dtype=np.float32) / 16000.0)

    chunks = processor.prepare_vq06_chunks(samples)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.waveform.shape == (16000,)
    assert chunk.features.shape == (128, 100)
    assert chunk.feature_length == 100
    assert chunk.duration_seconds == 1.0
    assert chunk.expected_token_length == 25


def test_cluster_linguistic_features_rejects_wrong_dim() -> None:
    processor = StepAudioTokenizerProcessor(_make_assets())
    with pytest.raises(ValueError):
        processor.cluster_linguistic_features(np.zeros((4, 3), dtype=np.float32))
