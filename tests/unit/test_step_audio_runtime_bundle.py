from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from mlx_speech.models.step_audio_editx import (
    STEP_AUDIO_EDITX_RUNTIME_FILES,
    StepAudioCosyVoiceMelConfig,
    validate_step_audio_editx_runtime_bundle,
)
from mlx_speech.models.step_audio_tokenizer import (
    RUNTIME_ASSETS_FILENAME,
    RUNTIME_CONFIG_FILENAME,
    StepAudioTokenizerConfig,
    StepAudioTokenizerProcessor,
    load_step_audio_tokenizer_runtime_assets,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "convert" / "step_audio_editx.py"
)


def _load_converter_module():
    spec = importlib.util.spec_from_file_location(
        "convert_step_audio_editx_runtime_bundle",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_assets(root: Path) -> tuple[np.ndarray, np.ndarray]:
    codebook = np.asarray(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        dtype=np.float32,
    )
    cmvn = np.asarray(
        [[-1.0, -2.0], [0.5, 0.25]],
        dtype=np.float32,
    )
    np.save(root / "linguistic_tokenizer.npy", codebook)
    (root / "speech_tokenizer_v1.onnx").touch()

    funasr = (
        root
        / "dengcunqin"
        / "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    )
    funasr.mkdir(parents=True)
    (funasr / "config.yaml").write_text(
        "model: ParaformerStreaming\n", encoding="utf-8"
    )
    (funasr / "model.pt").touch()
    (funasr / "am.mvn").write_text(
        "<AddShift> 2 2\n"
        "<LearnRateCoef> 0 [ -1.0 -2.0 ]\n"
        "<Rescale> 2 2\n"
        "<LearnRateCoef> 0 [ 0.5 0.25 ]\n",
        encoding="utf-8",
    )
    return codebook, cmvn


def test_converter_writes_self_contained_tokenizer_runtime_assets(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "bundle"
    source_dir.mkdir()
    output_dir.mkdir()
    expected_codebook, expected_cmvn = _write_source_assets(source_dir)

    converter = _load_converter_module()
    converter._save_tokenizer_runtime_assets(source_dir, output_dir)

    assert (output_dir / RUNTIME_ASSETS_FILENAME).is_file()
    assert (output_dir / RUNTIME_CONFIG_FILENAME).is_file()
    assert not (output_dir / "model.pt").exists()
    assert not (output_dir / "speech_tokenizer_v1.onnx").exists()

    runtime_config = json.loads(
        (output_dir / RUNTIME_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert set(runtime_config) == {
        "encoder_chunk_look_back",
        "model_type",
        "trim_frame_length",
        "trim_hop_length",
        "trim_keep_left_seconds",
        "trim_keep_right_seconds",
        "trim_output_hop_samples",
        "trim_top_db",
        "vq02_chunk_size",
        "vq02_codebook_size",
        "vq02_sample_rate",
        "vq06_hop_length",
        "vq06_max_chunk_seconds",
        "vq06_min_chunk_samples",
        "vq06_n_fft",
        "vq06_num_mels",
        "vq06_sample_rate",
        "vq06_token_rate_hz",
    }

    loaded = load_step_audio_tokenizer_runtime_assets(output_dir)
    assert loaded.model_dir == output_dir
    assert loaded.config.vq02_codebook_size == 3
    assert loaded.config.extra == {}
    np.testing.assert_array_equal(loaded.linguistic_codebook, expected_codebook)
    np.testing.assert_array_equal(loaded.cmvn, expected_cmvn)
    assert StepAudioTokenizerConfig.from_path(output_dir) == loaded.config
    processor = StepAudioTokenizerProcessor.from_path(output_dir)
    np.testing.assert_array_equal(
        processor.assets.linguistic_codebook, expected_codebook
    )


def test_converter_writes_frontend_runtime_config(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "bundle"
    cosyvoice_dir = source_dir / "CosyVoice-300M-25Hz"
    cosyvoice_dir.mkdir(parents=True)
    output_dir.mkdir()
    (cosyvoice_dir / "cosyvoice.yaml").write_text(
        "mel_conf:\n"
        "    num_mels: 80\n"
        "    n_fft: 1920\n"
        "    hop_size: 480\n"
        "    win_size: 1920\n"
        "    sampling_rate: 24000\n"
        "    fmin: 0\n"
        "    fmax: 8000\n",
        encoding="utf-8",
    )

    converter = _load_converter_module()
    converter._save_frontend_config(source_dir, output_dir)

    assert StepAudioCosyVoiceMelConfig.from_path(output_dir) == (
        StepAudioCosyVoiceMelConfig()
    )


def test_runtime_bundle_contract_is_strict(tmp_path: Path) -> None:
    for name in STEP_AUDIO_EDITX_RUNTIME_FILES:
        (tmp_path / name).touch()

    assert validate_step_audio_editx_runtime_bundle(tmp_path) == tmp_path

    (tmp_path / "vq06.safetensors").unlink()
    with np.testing.assert_raises_regex(FileNotFoundError, "vq06.safetensors"):
        validate_step_audio_editx_runtime_bundle(tmp_path)
