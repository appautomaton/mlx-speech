"""Tests for the Step-Audio vq02 runtime slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioVQ02Config,
    load_step_audio_vq02_checkpoint,
    load_step_audio_vq02_model,
)


MODEL_DIR = Path("models/stepfun/step_audio_tokenizer/original")
FUNASR_DIR = MODEL_DIR / "dengcunqin" / "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
HAS_ASSETS = FUNASR_DIR.exists()


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio FunASR assets not available")
def test_vq02_config_parses_local_yaml() -> None:
    config = StepAudioVQ02Config.from_config_yaml(FUNASR_DIR / "config.yaml")

    assert config.model_name == "ParaformerStreaming"
    assert config.frontend.sample_rate == 16000
    assert config.frontend.n_mels == 80
    assert config.frontend.lfr_m == 7
    assert config.frontend.lfr_n == 6
    assert config.encoder.input_size == 560
    assert config.encoder.output_size == 512
    assert config.encoder.num_blocks == 50
    assert config.encoder.attention_heads == 4


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio FunASR assets not available")
def test_vq02_checkpoint_filters_exact_encoder_state() -> None:
    checkpoint = load_step_audio_vq02_checkpoint(MODEL_DIR)

    assert len(checkpoint.state_dict) > 0
    assert all(key.startswith("encoder.") for key in checkpoint.state_dict)
    assert checkpoint.state_dict["encoder.encoders0.0.self_attn.fsmn_block.weight"].shape == (512, 11, 1)
    assert checkpoint.state_dict["encoder.encoders0.0.self_attn.linear_q_k_v.weight"].shape == (1536, 560)


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio FunASR assets not available")
def test_vq02_runtime_generates_token_ids_from_waveform() -> None:
    loaded = load_step_audio_vq02_model(MODEL_DIR)
    sample_count = 16000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_count, dtype=np.float32) / 16000.0)

    loaded.runtime.frontend.rng = np.random.default_rng(seed=0)
    tokens_first = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)
    loaded.runtime.frontend.rng = np.random.default_rng(seed=0)
    tokens_second = loaded.runtime.encode(audio, 16000, enable_trim=False, energy_norm=False)

    assert tokens_first == tokens_second
    assert len(tokens_first) > 0
    assert all(0 <= token < loaded.assets.config.vq02_codebook_size for token in tokens_first)


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio FunASR assets not available")
def test_vq02_runtime_encoder_features_match_cluster_length() -> None:
    loaded = load_step_audio_vq02_model(MODEL_DIR)
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
