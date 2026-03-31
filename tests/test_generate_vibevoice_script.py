from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_vibevoice.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_vibevoice_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    data = {
        "text": "Hello from VibeVoice.",
        "model_dir": None,
        "tokenizer_dir": None,
        "reference_audio": None,
        "output": "outputs/vibevoice_test.wav",
        "cfg_scale": 1.3,
        "diffusion_steps": 20,
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": None,
        "greedy": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_parser_help_documents_sampling_controls() -> None:
    module = _load_script_module()

    help_text = module._build_parser().format_help()

    assert "--temperature" in help_text
    assert "--top-p" in help_text
    assert "--seed" in help_text
    assert "--greedy" in help_text


def test_build_generation_config_uses_sampling_defaults() -> None:
    module = _load_script_module()

    config = module._build_generation_config(_args())

    assert config.do_sample is True
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.seed is None


def test_build_generation_config_can_force_greedy_with_seed() -> None:
    module = _load_script_module()

    config = module._build_generation_config(_args(greedy=True, seed=123, top_p=0.8))

    assert config.do_sample is False
    assert config.temperature == 0.0
    assert config.top_p == 0.8
    assert config.seed == 123
