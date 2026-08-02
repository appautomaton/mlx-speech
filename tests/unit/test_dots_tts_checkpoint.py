from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from pathlib import Path

from mlx_speech.models.dots_tts.checkpoint import (
    BASE_DTYPE_POLICY,
    _strict_load,
    align_state_dict,
    storage_dtype,
    storage_dtype_name,
)


def test_alignment_report_accepts_exact_native_bf16_state() -> None:
    model = nn.Linear(2, 3, bias=True)
    weights = {
        "weight": mx.zeros((3, 2), dtype=mx.bfloat16),
        "bias": mx.zeros((3,), dtype=mx.bfloat16),
    }
    report = align_state_dict(
        "tiny", model, weights, expected_dtype=mx.bfloat16
    )
    assert report.is_exact_match
    report.require_exact()


def test_alignment_report_accounts_for_all_failure_modes() -> None:
    model = nn.Linear(2, 3, bias=True)
    weights = {
        "weight": mx.zeros((2, 3), dtype=mx.float32),
        "unexpected": mx.zeros((1,), dtype=mx.bfloat16),
    }
    report = align_state_dict(
        "tiny", model, weights, expected_dtype=mx.bfloat16
    )
    assert report.missing_in_model == ("unexpected",)
    assert report.missing_in_checkpoint == ("bias",)
    assert report.shape_mismatches == (("weight", (3, 2), (2, 3)),)
    assert report.dtype_mismatches == (
        ("weight", "mlx.core.bfloat16", "mlx.core.float32"),
    )
    with pytest.raises(ValueError, match="alignment failed"):
        report.require_exact()


def test_alignment_report_rejects_duplicate_checkpoint_keys() -> None:
    model = nn.Linear(2, 3, bias=False)
    entries = [
        ("weight", mx.zeros((3, 2), dtype=mx.bfloat16)),
        ("weight", mx.ones((3, 2), dtype=mx.bfloat16)),
    ]
    report = align_state_dict(
        "tiny", model, entries, expected_dtype=mx.bfloat16
    )
    assert report.duplicate_checkpoint_keys == ("weight",)
    with pytest.raises(ValueError, match="duplicate_checkpoint_keys"):
        report.require_exact()


def test_base_dtype_policy_is_total_and_path_specific() -> None:
    assert storage_dtype_name(BASE_DTYPE_POLICY, "core", "qwen.weight") == (
        "bfloat16"
    )
    assert storage_dtype(
        BASE_DTYPE_POLICY, "vocoder", "audio_encoder.pre_conv.weight"
    ) == mx.float32
    assert storage_dtype(
        BASE_DTYPE_POLICY, "vocoder", "decoder.conv_pre.weight"
    ) == mx.bfloat16
    assert storage_dtype(BASE_DTYPE_POLICY, "speaker", "dense.weight") == (
        mx.float32
    )
    with pytest.raises(ValueError, match="exactly once"):
        storage_dtype_name(BASE_DTYPE_POLICY, "vocoder", "unknown.weight")


def test_strict_load_checks_runtime_dtype_after_binding(tmp_path: Path) -> None:
    class CastingLinear(nn.Linear):
        def load_weights(self, file_or_weights, strict: bool = True):
            super().load_weights(file_or_weights, strict=strict)
            self.set_dtype(mx.float32)

    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(
        str(path), {"weight": mx.ones((3, 2), dtype=mx.bfloat16)}
    )
    model = CastingLinear(2, 3, bias=False)
    with pytest.raises(ValueError, match="runtime_dtype_mismatches"):
        _strict_load(
            "tiny",
            model,
            path,
            expected_dtype=lambda key: mx.bfloat16,
        )
