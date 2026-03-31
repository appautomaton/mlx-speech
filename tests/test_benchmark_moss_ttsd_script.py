from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_moss_ttsd.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("benchmark_moss_ttsd_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_csv_choices_preserves_order_and_deduplicates() -> None:
    module = _load_script_module()

    values = module._parse_csv_choices(
        "medium,short,medium,long",
        valid={"short", "medium", "long"},
        argument_name="--prompts",
    )

    assert values == ["medium", "short", "long"]


def test_build_cases_orders_prompts_then_modes_then_weights_then_cache() -> None:
    module = _load_script_module()
    args = argparse.Namespace(
        prompts="short,medium",
        weights="quantized",
        cache="on,off",
        modes="greedy,sampled",
    )

    cases = module._build_cases(args)

    assert len(cases) == 8
    assert cases[0].prompt.name == "short"
    assert cases[0].sampling_mode == "greedy"
    assert cases[0].weights == "quantized"
    assert cases[0].use_kv_cache is True
    assert cases[-1].prompt.name == "medium"
    assert cases[-1].sampling_mode == "sampled"
    assert cases[-1].weights == "quantized"
    assert cases[-1].use_kv_cache is False


def test_build_generation_config_switches_greedy_and_sampling_modes() -> None:
    module = _load_script_module()
    short_prompt = module.PROMPT_SPECS["short"]

    greedy_case = module.BenchmarkCase(
        prompt=short_prompt,
        sampling_mode="greedy",
        use_kv_cache=True,
        weights="quantized",
    )
    sampled_case = module.BenchmarkCase(
        prompt=short_prompt,
        sampling_mode="sampled",
        use_kv_cache=False,
        weights="quantized",
    )

    greedy_config = module._build_generation_config(greedy_case)
    sampled_config = module._build_generation_config(sampled_case)

    assert greedy_config.use_kv_cache is True
    assert greedy_config.do_sample is False
    assert greedy_config.audio_temperature == 0.0
    assert sampled_config.use_kv_cache is False
    assert sampled_config.do_sample is True
    assert sampled_config.audio_temperature == 1.1
