"""Step-Audio runtime tests that require local checkpoints and run inference."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    StepAudioCosyVoiceMelConfig,
    load_step_audio_campplus_model,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
)
from tests.helpers.step_audio import EDITX_DIR, skip_no_cosyvoice

pytestmark = pytest.mark.runtime


def _sine_wave(sample_rate: int, frequency_hz: float, seconds: float = 1.0) -> np.ndarray:
    sample_count = int(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    return 0.1 * np.sin(2.0 * np.pi * frequency_hz * time)


@skip_no_cosyvoice
def test_frontend_config_parses_local_yaml() -> None:
    config = StepAudioCosyVoiceMelConfig.from_yaml_path(EDITX_DIR / "CosyVoice-300M-25Hz" / "cosyvoice.yaml")

    assert config.num_mels == 80
    assert config.n_fft == 1920
    assert config.hop_size == 480
    assert config.win_size == 1920
    assert config.sampling_rate == 24000
    assert config.fmin == 0.0
    assert config.fmax == 8000.0


@skip_no_cosyvoice
def test_frontend_extract_speech_feat_resamples_and_returns_prompt_shape() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    sample_rate = 16000
    audio = _sine_wave(sample_rate=sample_rate, frequency_hz=330.0)

    speech_feat, speech_feat_len = frontend.extract_speech_feat(audio, sample_rate)

    assert speech_feat.shape == (1, 50, 80)
    assert speech_feat_len.tolist() == [50]
    assert speech_feat.dtype == np.float32


@skip_no_cosyvoice
def test_campplus_runtime_extract_embedding_is_deterministic() -> None:
    loaded = load_step_audio_campplus_model(EDITX_DIR)
    audio = _sine_wave(sample_rate=16000, frequency_hz=440.0)

    first = loaded.runtime.extract_embedding(audio, 16000)
    second = loaded.runtime.extract_embedding(audio, 16000)

    assert first.shape == (1, 192)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


@skip_no_cosyvoice
def test_frontend_extract_spk_embedding_matches_loaded_runtime() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    direct = load_step_audio_campplus_model(EDITX_DIR)
    audio = _sine_wave(sample_rate=24000, frequency_hz=330.0)

    from_frontend = frontend.extract_spk_embedding(audio, 24000)
    from_runtime = direct.runtime.extract_embedding(audio, 24000)

    assert from_frontend.shape == (1, 192)
    assert from_frontend.dtype == np.float32
    assert np.allclose(from_frontend, from_runtime, atol=1e-5, rtol=1e-5)


@skip_no_cosyvoice
def test_flow_conditioner_prepares_nonstream_inputs_from_real_assets() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    conditioner = load_step_audio_flow_conditioner(EDITX_DIR)
    audio = _sine_wave(sample_rate=24000, frequency_hz=330.0)
    prompt_feat, _ = frontend.extract_speech_feat(audio, 24000)
    speaker_embedding = frontend.extract_spk_embedding(audio, 24000)

    prepared = conditioner.model.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029, 30, 31],
        prompt_token=[0, 1, 1024, 1025, 1026],
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


@skip_no_cosyvoice
def test_flow_model_inference_returns_generated_mel_shape() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    conditioner = load_step_audio_flow_conditioner(EDITX_DIR)
    flow_model = load_step_audio_flow_model(EDITX_DIR)
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


@skip_no_cosyvoice
def test_hift_inference_returns_waveform_shape_for_synthetic_mel() -> None:
    loaded = load_step_audio_hift_model(EDITX_DIR)
    mel = np.zeros((1, 80, 10), dtype=np.float32)

    waveform, source = loaded.model.inference(mel)

    assert waveform.shape == (1, 4800)
    assert source.shape == (1, 1, 4800)
    assert waveform.dtype == np.float32
    assert source.dtype == np.float32
    assert np.isfinite(waveform).all()
    assert np.isfinite(source).all()


@skip_no_cosyvoice
def test_flow_mel_output_decodes_to_waveform_via_hift() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    conditioner = load_step_audio_flow_conditioner(EDITX_DIR)
    flow_model = load_step_audio_flow_model(EDITX_DIR)
    hift_model = load_step_audio_hift_model(EDITX_DIR)

    sample_rate = 24000
    audio = _sine_wave(sample_rate=sample_rate, frequency_hz=330.0)
    prompt_feat, _ = frontend.extract_speech_feat(audio, sample_rate)
    speaker_embedding = frontend.extract_spk_embedding(audio, sample_rate)
    prepared = conditioner.model.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029, 30, 31],
        prompt_token=[0, 1, 1024, 1025, 1026],
        prompt_feat=prompt_feat,
        speaker_embedding=speaker_embedding,
    )

    mel = flow_model.model.inference(prepared, n_timesteps=2)
    waveform, source = hift_model.model.inference(mel)

    assert mel.shape == (1, 80, 12)
    assert waveform.shape == (1, 5760)
    assert source.shape == (1, 1, 5760)
    assert np.isfinite(waveform).all()
    assert np.isfinite(source).all()
