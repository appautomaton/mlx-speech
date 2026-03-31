from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_moss_sound_effect.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_moss_sound_effect_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    data = {
        "ambient_sound": "a sports car roaring past on the highway.",
        "duration_seconds": 10.0,
        "expected_tokens": None,
        "output": "out.wav",
        "model_dir": None,
        "codec_dir": None,
        "device": "auto",
        "max_new_tokens": 4096,
        "temperature": 1.5,
        "top_p": 0.6,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "greedy": False,
        "no_kv_cache": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_parser_help_documents_sound_effect_prompt_shape() -> None:
    module = _load_script_module()

    help_text = module._build_parser().format_help()

    assert "ambient sound description only" in help_text
    assert "12.5 tokens / second" in help_text
    assert "mlx-4bit" in help_text


def test_resolve_device_name_defaults_to_gpu() -> None:
    module = _load_script_module()

    assert module._resolve_device_name("auto") == "gpu"
    assert module._resolve_device_name("gpu") == "gpu"
    assert module._resolve_device_name("cpu") == "cpu"


def test_build_generation_config_uses_upstream_sound_effect_defaults() -> None:
    module = _load_script_module()

    config = module._build_generation_config(_args())

    assert config.use_kv_cache is True
    assert config.do_sample is True
    assert config.audio_temperature == 1.5
    assert config.audio_top_p == 0.6
    assert config.audio_top_k == 50
    assert config.audio_repetition_penalty == 1.2


def test_build_generation_config_can_disable_cache_and_force_greedy() -> None:
    module = _load_script_module()

    config = module._build_generation_config(
        _args(greedy=True, no_kv_cache=True)
    )

    assert config.use_kv_cache is False
    assert config.do_sample is False
    assert config.audio_temperature == 0.0
