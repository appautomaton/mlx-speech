"""Step-Audio tokenizer runtime tests that require local checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_speech.models.step_audio_tokenizer import (
    load_step_audio_vq02_model,
    load_step_audio_vq06_model,
)
from tests.helpers.step_audio import TOKENIZER_DIR, skip_no_funasr, skip_no_vq06

pytestmark = pytest.mark.runtime


@skip_no_funasr
def test_vq02_runtime_generates_token_ids_from_waveform() -> None:
    loaded = load_step_audio_vq02_model(TOKENIZER_DIR)
    sample_count = 16000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_count, dtype=np.float32) / 16000.0)

    loaded.runtime.frontend.rng = np.random.default_rng(seed=0)
    tokens_first = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)
    loaded.runtime.frontend.rng = np.random.default_rng(seed=0)
    tokens_second = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert tokens_first == tokens_second
    assert len(tokens_first) > 0
    assert all(0 <= token < loaded.assets.config.vq02_codebook_size for token in tokens_first)


@skip_no_funasr
def test_vq02_runtime_encoder_features_match_cluster_length() -> None:
    loaded = load_step_audio_vq02_model(TOKENIZER_DIR)
    audio = 0.05 * np.sin(2.0 * np.pi * 330.0 * np.arange(12000, dtype=np.float32) / 16000.0)
    waveform = loaded.runtime.processor.preprocess_wav(
        audio,
        16000,
        enable_trim=False,
        energy_norm=False,
    )

    loaded.runtime.frontend.rng = np.random.default_rng(seed=0)
    features = loaded.runtime.extract_encoder_features(waveform, is_final=True)
    tokens = loaded.runtime.processor.cluster_linguistic_features(features)

    assert features.ndim == 2
    assert features.shape[1] == loaded.config.encoder.output_size
    assert len(tokens) == features.shape[0]


@skip_no_vq06
def test_vq06_runtime_generates_token_ids_from_waveform() -> None:
    loaded = load_step_audio_vq06_model(TOKENIZER_DIR)
    sample_count = 16000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_count, dtype=np.float32) / 16000.0)

    tokens_first = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)
    tokens_second = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert tokens_first == tokens_second
    assert len(tokens_first) > 0
    assert all(0 <= token < loaded.config.codebook_size for token in tokens_first)


@skip_no_vq06
def test_vq06_runtime_token_length_matches_chunk_expectation() -> None:
    loaded = load_step_audio_vq06_model(TOKENIZER_DIR)
    audio = 0.05 * np.sin(2.0 * np.pi * 330.0 * np.arange(16000, dtype=np.float32) / 16000.0)
    waveform = loaded.runtime.processor.preprocess_wav(
        audio,
        16000,
        enable_trim=False,
        energy_norm=False,
    )

    chunks = loaded.runtime.processor.prepare_vq06_chunks(waveform)
    assert len(chunks) == 1

    token_ids = loaded.runtime.encode_chunk(chunks[0].features)

    assert abs(len(token_ids) - chunks[0].expected_token_length) <= 2


@skip_no_vq06
def test_vq06_runtime_concatenates_multiple_chunks() -> None:
    loaded = load_step_audio_vq06_model(TOKENIZER_DIR)
    seconds = 31
    audio = 0.05 * np.sin(2.0 * np.pi * 180.0 * np.arange(seconds * 16000, dtype=np.float32) / 16000.0)

    token_ids = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert 770 <= len(token_ids) <= 780
    assert all(0 <= token < loaded.config.codebook_size for token in token_ids)
