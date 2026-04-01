"""Tests for the Step-Audio non-stream flow conditioning slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    interpolate_prompt_features,
    load_step_audio_flow_model,
    load_step_audio_flow_conditioner,
    reshape_mixed_audio_tokens,
)


MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
COSYVOICE_DIR = MODEL_DIR / "CosyVoice-300M-25Hz"
HAS_ASSETS = COSYVOICE_DIR.exists()


def _sine_wave(sample_rate: int, frequency_hz: float, seconds: float = 1.0) -> np.ndarray:
    sample_count = int(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    return 0.1 * np.sin(2.0 * np.pi * frequency_hz * time)


def test_reshape_mixed_audio_tokens_matches_shipped_grouping() -> None:
    tokens = [10, 11, 1024, 1025, 1026]

    reshaped = reshape_mixed_audio_tokens(tokens)

    expected = np.asarray(
        [
            [10, 1025],
            [11, 1026],
            [1024, 1027],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(reshaped, expected)


def test_reshape_mixed_audio_tokens_matches_shipped_padding_behavior() -> None:
    tokens = [10, 11, 1024, 1025, 1026, 20, 21]

    reshaped = reshape_mixed_audio_tokens(tokens)

    expected = np.asarray(
        [
            [10, 1025],
            [11, 1026],
            [1024, 1027],
            [20, 1025],
            [21, 1025],
            [1024, 1025],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(reshaped, expected)


def test_interpolate_prompt_features_uses_nearest_neighbor_time_repeat() -> None:
    prompt_feat = np.asarray([[[1.0], [3.0]]], dtype=np.float32)

    interpolated = interpolate_prompt_features(prompt_feat, target_length=6)

    expected = np.asarray([[[1.0], [1.0], [1.0], [3.0], [3.0], [3.0]]], dtype=np.float32)
    assert np.array_equal(interpolated, expected)


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_flow_conditioner_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_flow_conditioner(MODEL_DIR)

    assert loaded.alignment_report.is_exact_match
    assert loaded.config.vocab_size == 5121
    assert loaded.config.input_size == 512
    assert loaded.config.output_size == 80
    assert loaded.config.spk_embed_dim == 192


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_flow_model_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_flow_model(MODEL_DIR)

    assert loaded.alignment_report.is_exact_match
    assert loaded.config.input_size == 512
    assert loaded.config.output_size == 80
    assert loaded.config.estimator_depth == 16
    assert loaded.config.num_blocks == 6
    assert loaded.config.num_up_blocks == 4


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_flow_conditioner_prepares_nonstream_inputs_from_real_assets() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(MODEL_DIR)
    conditioner = load_step_audio_flow_conditioner(MODEL_DIR)
    audio = _sine_wave(sample_rate=24000, frequency_hz=330.0)
    prompt_feat, _ = frontend.extract_speech_feat(audio, 24000)
    speaker_embedding = frontend.extract_spk_embedding(audio, 24000)

    prompt_token = [0, 1, 1024, 1025, 1026]
    token = [20, 21, 1027, 1028, 1029, 30, 31]
    prepared = conditioner.model.prepare_nonstream_inputs(
        token=token,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        speaker_embedding=speaker_embedding,
    )

    assert prepared.prompt_token_dual.shape == (1, 3, 2)
    assert prepared.token_dual.shape == (1, 6, 2)
    assert prepared.concatenated_token_dual.shape == (1, 9, 2)
    assert prepared.concatenated_token_length.tolist() == [9]
    assert prepared.embedded_tokens.shape == (1, 9, 512)
    assert prepared.prompt_feat_aligned.shape == (1, 6, 80)
    assert prepared.projected_speaker_embedding.shape == (1, 80)
    assert np.allclose(
        np.linalg.norm(prepared.normalized_speaker_embedding, axis=1),
        np.ones((1,), dtype=np.float32),
        atol=1e-5,
    )


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_flow_model_inference_returns_generated_mel_shape() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(MODEL_DIR)
    conditioner = load_step_audio_flow_conditioner(MODEL_DIR)
    flow_model = load_step_audio_flow_model(MODEL_DIR)
    audio = _sine_wave(sample_rate=24000, frequency_hz=330.0)
    prompt_feat, _ = frontend.extract_speech_feat(audio, 24000)
    speaker_embedding = frontend.extract_spk_embedding(audio, 24000)

    prepared = conditioner.model.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029, 30, 31],
        prompt_token=[0, 1, 1024, 1025, 1026],
        prompt_feat=prompt_feat,
        speaker_embedding=speaker_embedding,
    )
    mel = flow_model.model.inference(prepared, n_timesteps=2)

    assert mel.shape == (1, 80, 12)
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()
