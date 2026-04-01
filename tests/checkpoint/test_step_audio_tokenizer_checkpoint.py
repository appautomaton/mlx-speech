"""Step-Audio tokenizer asset and checkpoint validation tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioVQ02Config,
    StepAudioVQ06Model,
    load_step_audio_funasr_checkpoint,
    load_step_audio_tokenizer_assets,
    load_step_audio_vq02_checkpoint,
    load_step_audio_vq06_checkpoint,
    validate_step_audio_vq06_checkpoint_against_model,
)
from tests.helpers.step_audio import (
    EDITX_DIR,
    FUNASR_DIR,
    HAS_LOCAL_TOKENIZER,
    TOKENIZER_DIR,
    skip_no_editx,
    skip_no_funasr,
    skip_no_tokenizer,
    skip_no_vq06,
)

TOKENIZER_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "mlx_speech" / "models" / "step_audio_editx" / "tokenizer.py"

pytestmark = pytest.mark.checkpoint


@skip_no_tokenizer
def test_load_step_audio_tokenizer_assets_from_local_dir() -> None:
    loaded = load_step_audio_tokenizer_assets(TOKENIZER_DIR)

    assert loaded.model_dir == TOKENIZER_DIR
    assert loaded.config.vq02_codebook_size == 1024
    assert loaded.config.vq06_codebook_size == 4096
    assert loaded.config.vq06_n_fft == 400
    assert loaded.config.vq06_hop_length == 160
    assert loaded.config.vq02_chunk_size == (0, 4, 5)
    assert loaded.linguistic_tokenizer_path == TOKENIZER_DIR / "linguistic_tokenizer.npy"
    assert loaded.semantic_tokenizer_path == TOKENIZER_DIR / "speech_tokenizer_v1.onnx"
    assert loaded.funasr_model_dir.name == "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    assert loaded.funasr_config_path.exists()
    assert loaded.funasr_checkpoint_path.exists()
    assert loaded.linguistic_codebook.shape[0] == 1024


@skip_no_funasr
def test_load_step_audio_funasr_checkpoint_from_local_dir() -> None:
    checkpoint = load_step_audio_funasr_checkpoint(TOKENIZER_DIR)

    assert checkpoint.checkpoint_path.name == "model.pt"
    assert "pytorch_model/data.pkl" in checkpoint.files
    assert "encoder.encoders0.0.self_attn.linear_out.weight" in checkpoint.weights
    assert "encoder.after_norm.weight" in checkpoint.weights
    assert checkpoint.weights["encoder.encoders0.0.self_attn.linear_out.weight"].shape == (512, 512)
    assert checkpoint.weights["encoder.encoders0.0.self_attn.linear_q_k_v.weight"].shape == (1536, 560)
    assert checkpoint.weights["encoder.after_norm.weight"].shape == (512,)


@skip_no_funasr
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


@skip_no_funasr
def test_vq02_checkpoint_filters_exact_encoder_state() -> None:
    checkpoint = load_step_audio_vq02_checkpoint(TOKENIZER_DIR)

    assert len(checkpoint.state_dict) > 0
    assert all(key.startswith("encoder.") for key in checkpoint.state_dict)
    assert checkpoint.state_dict["encoder.encoders0.0.self_attn.fsmn_block.weight"].shape == (512, 11, 1)
    assert checkpoint.state_dict["encoder.encoders0.0.self_attn.linear_q_k_v.weight"].shape == (1536, 560)


@skip_no_vq06
def test_vq06_checkpoint_parses_local_onnx_config() -> None:
    checkpoint = load_step_audio_vq06_checkpoint(TOKENIZER_DIR)

    assert checkpoint.config.num_mels == 128
    assert checkpoint.config.hidden_size == 1280
    assert checkpoint.config.num_heads == 20
    assert checkpoint.config.num_layers == 6
    assert checkpoint.config.max_positions == 1500
    assert checkpoint.config.codebook_size == 4096
    assert checkpoint.state_dict["encoder.conv1.weight"].shape == (1280, 3, 128)
    assert checkpoint.state_dict["quantizer.codebook"].shape == (1280, 4096)


@skip_no_vq06
def test_vq06_checkpoint_aligns_exact_model_state() -> None:
    checkpoint = load_step_audio_vq06_checkpoint(TOKENIZER_DIR)
    model = StepAudioVQ06Model(checkpoint.config)

    report = validate_step_audio_vq06_checkpoint_against_model(model, checkpoint)

    assert report.is_exact_match
    assert report.missing_in_model == ()
    assert report.missing_in_checkpoint == ()
    assert report.shape_mismatches == ()


@skip_no_editx
@pytest.mark.skipif(not HAS_LOCAL_TOKENIZER, reason="Step-Audio tokenizer.json not found")
def test_real_tokenizer_json_handles_special_tokens_as_single_ids() -> None:
    spec = importlib.util.spec_from_file_location(
        "step_audio_editx_tokenizer_checkpoint_module",
        TOKENIZER_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tokenizer module from {TOKENIZER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    tokenizer = module.StepAudioEditXTokenizer.from_path(EDITX_DIR)

    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.token_to_id("<|EOT|>") == 3
    assert tokenizer.token_to_id("<|BOT|>") == 4
    assert tokenizer.encode("<s><|BOT|> assistant\n<audio_1><|EOT|>") == [1, 4, 15886, 78, 65537, 3]
