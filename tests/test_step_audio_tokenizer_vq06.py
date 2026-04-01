"""Tests for the Step-Audio vq06 runtime slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioVQ06Model,
    load_step_audio_vq06_checkpoint,
    load_step_audio_vq06_model,
    validate_step_audio_vq06_checkpoint_against_model,
)


MODEL_DIR = Path("models/stepfun/step_audio_tokenizer/original")
HAS_ASSETS = (MODEL_DIR / "speech_tokenizer_v1.onnx").exists()


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio semantic tokenizer assets not available")
def test_vq06_checkpoint_parses_local_onnx_config() -> None:
    checkpoint = load_step_audio_vq06_checkpoint(MODEL_DIR)

    assert checkpoint.config.num_mels == 128
    assert checkpoint.config.hidden_size == 1280
    assert checkpoint.config.num_heads == 20
    assert checkpoint.config.num_layers == 6
    assert checkpoint.config.max_positions == 1500
    assert checkpoint.config.codebook_size == 4096
    assert checkpoint.state_dict["encoder.conv1.weight"].shape == (1280, 3, 128)
    assert checkpoint.state_dict["quantizer.codebook"].shape == (1280, 4096)


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio semantic tokenizer assets not available")
def test_vq06_checkpoint_aligns_exact_model_state() -> None:
    checkpoint = load_step_audio_vq06_checkpoint(MODEL_DIR)
    model = StepAudioVQ06Model(checkpoint.config)

    report = validate_step_audio_vq06_checkpoint_against_model(model, checkpoint)

    assert report.is_exact_match
    assert report.missing_in_model == ()
    assert report.missing_in_checkpoint == ()
    assert report.shape_mismatches == ()


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio semantic tokenizer assets not available")
def test_vq06_runtime_generates_token_ids_from_waveform() -> None:
    loaded = load_step_audio_vq06_model(MODEL_DIR)
    sample_count = 16000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_count, dtype=np.float32) / 16000.0)

    tokens_first = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)
    tokens_second = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert tokens_first == tokens_second
    assert len(tokens_first) > 0
    assert all(0 <= token < loaded.config.codebook_size for token in tokens_first)


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio semantic tokenizer assets not available")
def test_vq06_runtime_token_length_matches_chunk_expectation() -> None:
    loaded = load_step_audio_vq06_model(MODEL_DIR)
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


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio semantic tokenizer assets not available")
def test_vq06_runtime_concatenates_multiple_chunks() -> None:
    loaded = load_step_audio_vq06_model(MODEL_DIR)
    seconds = 31
    audio = 0.05 * np.sin(2.0 * np.pi * 180.0 * np.arange(seconds * 16000, dtype=np.float32) / 16000.0)

    token_ids = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert len(token_ids) == 775
    assert all(0 <= token < loaded.config.codebook_size for token in token_ids)
