"""Tests for the Step-Audio dual tokenizer family."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioTokenizerConfig,
    deinterleave_step_audio_tokens,
    format_audio_token_string,
    interleave_step_audio_tokens,
    load_step_audio_funasr_checkpoint,
    load_step_audio_tokenizer_assets,
    mixed_ids_to_prompt_tokens,
    pack_raw_codes_to_mixed_ids,
    pack_raw_codes_to_prompt_tokens,
    parse_audio_token_string,
    prompt_tokens_to_audio_token_string,
    prompt_tokens_to_mixed_ids,
    raw_vq02_to_mixed_ids,
    raw_vq06_to_mixed_ids,
    resolve_step_audio_tokenizer_model_dir,
    unpack_mixed_ids_to_raw_codes,
    unpack_prompt_tokens_to_raw_codes,
)

MODEL_DIR = Path("models/stepfun/step_audio_tokenizer/original")
HAS_ASSETS = MODEL_DIR.exists()


def test_package_is_importable() -> None:
    cfg = StepAudioTokenizerConfig()
    assert cfg.model_type == "step_audio_tokenizer"
    assert cfg.group_size == 5
    assert cfg.prompt_audio_vocab_size == 5120


def test_interleave_and_deinterleave_round_trip() -> None:
    vq02 = [1, 2, 3, 4]
    vq06 = [1034, 1035, 1036, 1037, 1038, 1039]

    interleaved = interleave_step_audio_tokens(vq02, vq06)
    assert interleaved == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]

    roundtrip_vq02, roundtrip_vq06 = deinterleave_step_audio_tokens(interleaved)
    assert roundtrip_vq02 == vq02
    assert roundtrip_vq06 == vq06


def test_raw_prompt_and_mixed_offset_math() -> None:
    assert raw_vq02_to_mixed_ids([0, 1, 2]) == [65536, 65537, 65538]
    assert raw_vq06_to_mixed_ids([0, 1, 2]) == [66560, 66561, 66562]
    assert prompt_tokens_to_mixed_ids([0, 1, 1024, 1025]) == [65536, 65537, 66560, 66561]
    assert mixed_ids_to_prompt_tokens([65536, 65537, 66560, 66561]) == [0, 1, 1024, 1025]


def test_pack_and_unpack_raw_codes() -> None:
    vq02_raw = [1, 2, 3, 4]
    vq06_raw = [10, 11, 12, 13, 14, 15]

    prompt_tokens = pack_raw_codes_to_prompt_tokens(vq02_raw, vq06_raw)
    assert prompt_tokens == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]
    assert unpack_prompt_tokens_to_raw_codes(prompt_tokens) == (vq02_raw, vq06_raw)

    mixed_ids = pack_raw_codes_to_mixed_ids(vq02_raw, vq06_raw)
    assert mixed_ids == [65537, 65538, 66570, 66571, 66572, 65539, 65540, 66573, 66574, 66575]
    assert unpack_mixed_ids_to_raw_codes(mixed_ids) == (vq02_raw, vq06_raw)


def test_audio_token_string_format_and_parse() -> None:
    text = format_audio_token_string(
        [1, 2, 3, 4],
        [10, 11, 12, 13, 14, 15],
    )
    assert text == (
        "<audio_1><audio_2><audio_1034><audio_1035><audio_1036>"
        "<audio_3><audio_4><audio_1037><audio_1038><audio_1039>"
    )
    prompt_tokens = parse_audio_token_string(text)
    assert prompt_tokens == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]
    assert prompt_tokens_to_audio_token_string(prompt_tokens) == text


def test_deinterleave_truncates_incomplete_group_by_default() -> None:
    vq02, vq06 = deinterleave_step_audio_tokens([1, 2, 1034, 1035, 1036, 9])
    assert vq02 == [1, 2]
    assert vq06 == [1034, 1035, 1036]

    with pytest.raises(ValueError):
        deinterleave_step_audio_tokens([1, 2, 1034, 1035, 1036, 9], strict=True)


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio tokenizer assets not available")
def test_load_step_audio_tokenizer_assets_from_local_dir() -> None:
    assert resolve_step_audio_tokenizer_model_dir() == MODEL_DIR.resolve()
    loaded = load_step_audio_tokenizer_assets(MODEL_DIR)

    assert loaded.model_dir == MODEL_DIR
    assert loaded.config.vq02_codebook_size == 1024
    assert loaded.config.vq06_codebook_size == 4096
    assert loaded.config.vq06_n_fft == 400
    assert loaded.config.vq06_hop_length == 160
    assert loaded.config.vq02_chunk_size == (0, 4, 5)
    assert loaded.linguistic_tokenizer_path == MODEL_DIR / "linguistic_tokenizer.npy"
    assert loaded.semantic_tokenizer_path == MODEL_DIR / "speech_tokenizer_v1.onnx"
    assert loaded.funasr_model_dir.name == "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    assert loaded.funasr_config_path.exists()
    assert loaded.funasr_checkpoint_path.exists()
    assert loaded.linguistic_codebook.shape[0] == 1024


@pytest.mark.skipif(not HAS_ASSETS, reason="Step-Audio tokenizer assets not available")
def test_load_step_audio_funasr_checkpoint_from_local_dir() -> None:
    checkpoint = load_step_audio_funasr_checkpoint(MODEL_DIR)

    assert checkpoint.checkpoint_path.name == "model.pt"
    assert "pytorch_model/data.pkl" in checkpoint.files
    assert "encoder.encoders0.0.self_attn.linear_out.weight" in checkpoint.weights
    assert "encoder.after_norm.weight" in checkpoint.weights
    assert checkpoint.weights["encoder.encoders0.0.self_attn.linear_out.weight"].shape == (512, 512)
    assert checkpoint.weights["encoder.encoders0.0.self_attn.linear_q_k_v.weight"].shape == (1536, 560)
    assert checkpoint.weights["encoder.after_norm.weight"].shape == (512,)
