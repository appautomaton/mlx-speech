"""Step-Audio checkpoint and config validation tests."""

from __future__ import annotations

import pytest

from mlx_speech.models.step_audio_editx import (
    Step1Config,
    Step1ForCausalLM,
    StepAudioCampPlusConfig,
    load_step_audio_campplus_model,
    load_step_audio_editx_checkpoint,
    load_step_audio_editx_model,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
    resolve_step_audio_cosyvoice_dir,
    validate_checkpoint_against_model,
)
from tests.helpers.step_audio import EDITX_DIR, COSYVOICE_DIR, skip_no_cosyvoice, skip_no_editx

pytestmark = pytest.mark.checkpoint


@skip_no_editx
def test_step1_config_parses_shipped_config() -> None:
    cfg = Step1Config.from_path(EDITX_DIR)
    assert cfg.hidden_size == 3072
    assert cfg.intermediate_size == 8192
    assert cfg.num_attention_heads == 48
    assert cfg.num_attention_groups == 4
    assert cfg.num_hidden_layers == 32
    assert cfg.vocab_size == 74752
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    assert cfg.bos_token_id == 1
    assert cfg.pad_token_id == 0
    assert cfg.eos_token_id == 3
    assert cfg.head_dim == 64
    assert cfg.kv_repeat == 12


@skip_no_editx
def test_step1_checkpoint_alignment_is_exact() -> None:
    checkpoint = load_step_audio_editx_checkpoint(EDITX_DIR)
    model = Step1ForCausalLM(checkpoint.config)
    report = validate_checkpoint_against_model(model, checkpoint)
    assert report.is_exact_match
    assert checkpoint.key_count == 291


@skip_no_editx
def test_step1_model_loads_real_checkpoint() -> None:
    loaded = load_step_audio_editx_model(EDITX_DIR, strict=True, prefer_mlx_int8=False)
    assert loaded.alignment_report.is_exact_match
    assert loaded.model.config.hidden_size == 3072


@skip_no_cosyvoice
def test_cosyvoice_dir_resolves_from_step_audio_root() -> None:
    assert resolve_step_audio_cosyvoice_dir(EDITX_DIR) == COSYVOICE_DIR
    assert resolve_step_audio_cosyvoice_dir(COSYVOICE_DIR) == COSYVOICE_DIR


@skip_no_cosyvoice
def test_campplus_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_campplus_model(EDITX_DIR)
    assert loaded.alignment_report.is_exact_match
    assert loaded.config.embedding_size == 192
    assert loaded.config.block_layers == (12, 24, 16)


@skip_no_cosyvoice
def test_flow_conditioner_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_flow_conditioner(EDITX_DIR)
    assert loaded.alignment_report.is_exact_match
    assert loaded.config.vocab_size == 5121
    assert loaded.config.input_size == 512
    assert loaded.config.output_size == 80
    assert loaded.config.spk_embed_dim == 192


@skip_no_cosyvoice
def test_flow_model_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_flow_model(EDITX_DIR)
    assert loaded.alignment_report.is_exact_match
    assert loaded.config.input_size == 512
    assert loaded.config.output_size == 80
    assert loaded.config.estimator_depth == 16
    assert loaded.config.num_blocks == 6
    assert loaded.config.num_up_blocks == 4


@skip_no_cosyvoice
def test_hift_config_parses_local_yaml_and_alignment_is_exact() -> None:
    config = StepAudioCampPlusConfig  # keep import live for checkpoint tier parity with other model tests
    _ = config
    loaded = load_step_audio_hift_model(EDITX_DIR)
    assert loaded.alignment_report.is_exact_match
    assert loaded.config.sampling_rate == 24000
    assert loaded.config.upsample_rates == (8, 5, 3)
    assert loaded.config.istft_n_fft == 16
    assert loaded.config.istft_hop_len == 4
