from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_moss_ttsd.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_moss_ttsd_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _single_args(**overrides):
    data = {
        "mode": "generation",
        "text": "[S1] Watson, we should leave now.",
        "output": "out.wav",
        "input_jsonl": None,
        "save_dir": None,
        "prompt_audio_speaker1": None,
        "prompt_text_speaker1": None,
        "prompt_audio_speaker2": None,
        "prompt_text_speaker2": None,
        "prompt_audio_speaker3": None,
        "prompt_text_speaker3": None,
        "prompt_audio_speaker4": None,
        "prompt_text_speaker4": None,
        "prompt_audio_speaker5": None,
        "prompt_text_speaker5": None,
        "text_normalize": False,
        "sample_rate_normalize": False,
        "max_new_tokens": 96,
        "temperature": 1.1,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "greedy": False,
        "no_kv_cache": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_resolve_device_name_defaults_to_gpu() -> None:
    module = _load_script_module()

    assert module._resolve_device_name("auto") == "gpu"
    assert module._resolve_device_name("gpu") == "gpu"
    assert module._resolve_device_name("cpu") == "cpu"


def test_parser_help_documents_modes_and_batch_jsonl_shape() -> None:
    module = _load_script_module()

    help_text = module._build_parser().format_help()

    assert "Mode requirements:" in help_text
    assert "voice_clone_and_continuation" in help_text
    assert "Batch JSONL fields:" in help_text
    assert "prompt_audio_speakerN" in help_text
    assert "mlx-int8" in help_text
    assert "--prefer-original" not in help_text


def test_resolve_io_mode_distinguishes_single_and_batch() -> None:
    module = _load_script_module()

    assert module._resolve_io_mode(_single_args()) == "single"
    assert module._resolve_io_mode(
        _single_args(text=None, output=None, input_jsonl="batch.jsonl", save_dir="outputs")
    ) == "batch"


def test_validate_args_accepts_single_sample_shape() -> None:
    module = _load_script_module()

    module._validate_args(_single_args())


def test_validate_args_rejects_missing_single_text() -> None:
    module = _load_script_module()

    try:
        module._validate_args(_single_args(text=None))
    except ValueError as exc:
        assert "--text" in str(exc)
    else:
        raise AssertionError("Expected missing single-sample text to fail.")


def test_validate_args_accepts_batch_shape() -> None:
    module = _load_script_module()

    module._validate_args(
        _single_args(text=None, output=None, input_jsonl="batch.jsonl", save_dir="outputs")
    )


def test_validate_args_rejects_voice_clone_without_reference_audio() -> None:
    module = _load_script_module()

    try:
        module._validate_args(_single_args(mode="voice_clone"))
    except ValueError as exc:
        assert "--prompt-audio-speakerN" in str(exc)
    else:
        raise AssertionError("Expected missing voice_clone reference audio to fail.")


def test_validate_args_rejects_continuation_without_paired_prompt_fields() -> None:
    module = _load_script_module()

    try:
        module._validate_args(
            _single_args(
                mode="continuation",
                prompt_audio_speaker1="speaker.wav",
                prompt_text_speaker1=None,
            )
        )
    except ValueError as exc:
        assert "--prompt-audio-speakerN" in str(exc)
        assert "--prompt-text-speakerN" in str(exc)
    else:
        raise AssertionError("Expected missing paired continuation prompt fields to fail.")


def test_collect_single_sample_fields_keeps_speaker_prompt_fields() -> None:
    module = _load_script_module()
    args = _single_args(
        prompt_audio_speaker1="s1.wav",
        prompt_text_speaker1="[S1] Hello there.",
        prompt_audio_speaker2="s2.wav",
        prompt_text_speaker2="[S2] Good evening.",
    )

    sample = module._collect_single_sample_fields(args)

    assert sample["text"] == "[S1] Watson, we should leave now."
    assert sample["prompt_audio_speaker1"] == "s1.wav"
    assert sample["prompt_text_speaker2"] == "[S2] Good evening."


def test_build_generation_config_defaults_to_kv_cache_on() -> None:
    module = _load_script_module()

    config = module._build_generation_config(_single_args())

    assert config.use_kv_cache is True
    assert config.do_sample is True
    assert config.audio_temperature == 1.1


def test_build_generation_config_can_disable_kv_cache_and_force_greedy() -> None:
    module = _load_script_module()

    config = module._build_generation_config(
        _single_args(greedy=True, no_kv_cache=True)
    )

    assert config.use_kv_cache is False
    assert config.do_sample is False
    assert config.audio_temperature == 0.0
