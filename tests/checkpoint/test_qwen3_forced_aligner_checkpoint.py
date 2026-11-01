"""Strict loading of the local Qwen3-ForcedAligner packages.

Tier-2: needs ``models/qwen3_forced_aligner_0_6b/{original,mlx-bf16,mlx-int8}``;
each package skips when absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.nn as nn
import pytest
from safetensors import safe_open

from mlx_speech.models.qwen3_asr.checkpoint import (
    get_quantization_config,
    load_checkpoint_into_model,
    load_qwen3_asr_checkpoint,
    quantize_qwen3_asr_model,
)
from mlx_speech.models.qwen3_asr.forced_aligner import Qwen3ForcedAlignerSettings
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel


ROOT = Path("models/qwen3_forced_aligner_0_6b")
PACKAGES = ("original", "mlx-bf16", "mlx-int8")
GENERATED = {"audio_tower.positional_embedding.positional_embedding"}

pytestmark = pytest.mark.checkpoint


def _require(package: str) -> Path:
    model_dir = ROOT / package
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip(f"Qwen3-ForcedAligner {package} package not present: {model_dir}")
    return model_dir


@pytest.mark.parametrize("package", PACKAGES)
def test_package_loads_strictly_with_a_bf16_time_head(package):
    model_dir = _require(package)
    checkpoint = load_qwen3_asr_checkpoint(model_dir)
    settings = Qwen3ForcedAlignerSettings.from_config(checkpoint.config)
    assert settings == Qwen3ForcedAlignerSettings(151705, 80.0, 5000)

    model = Qwen3ASRModel(checkpoint.config)
    quantization = get_quantization_config(checkpoint.config)
    if quantization is not None:
        quantize_qwen3_asr_model(model, quantization, state_dict=checkpoint.state_dict)
    report = load_checkpoint_into_model(model, checkpoint, strict=True)

    assert not report.checkpoint_only
    assert not report.shape_mismatches
    assert set(report.model_only) <= GENERATED
    head = model.text_decoder.lm_head
    assert type(head) is nn.Linear
    assert head.weight.shape == (5000, 1024)


def test_original_and_bf16_store_708_bf16_tensors():
    for package in ("original", "mlx-bf16"):
        with safe_open(_require(package) / "model.safetensors", framework="numpy") as handle:
            keys = list(handle.keys())
            assert len(keys) == 708
            assert {handle.get_slice(key).get_dtype() for key in keys} == {"BF16"}


def test_int8_quantizes_everything_but_the_time_head():
    model_dir = _require("mlx-int8")
    checkpoint = load_qwen3_asr_checkpoint(model_dir)
    quantization = get_quantization_config(checkpoint.config)
    assert quantization is not None
    assert quantization.to_dict() == {"bits": 8, "group_size": 64, "mode": "affine"}

    with safe_open(model_dir / "model.safetensors", framework="numpy") as handle:
        keys = set(handle.keys())
        head = handle.get_slice("text_decoder.lm_head.weight")
        assert (head.get_shape(), head.get_dtype()) == ([5000, 1024], "BF16")
    scales = {key.removesuffix(".scales") for key in keys if key.endswith(".scales")}
    assert "text_decoder.lm_head" not in scales
    assert "text_decoder.model.embed_tokens" in scales
    assert sum(key.startswith("audio_tower.") for key in scales) == 24 * 6 + 3
    assert sum(key.startswith("text_decoder.model.layers.") for key in scales) == 28 * 7
