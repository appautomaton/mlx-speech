"""LongCat AudioDiT runtime tests that require local MLX runtime assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.longcat_audiodit.checkpoint import load_longcat_model

pytestmark = pytest.mark.runtime

LONGCAT_INT8_DIR = Path("models/longcat_audiodit/mlx-int8")
skip_no_longcat_int8 = pytest.mark.skipif(
    not (LONGCAT_INT8_DIR / "model.safetensors").exists(),
    reason="No local LongCat mlx-int8 runtime assets available",
)


@skip_no_longcat_int8
def test_longcat_quantized_model_loads_local_runtime_bundle() -> None:
    loaded = load_longcat_model(LONGCAT_INT8_DIR, prefer_mlx_int8=True, strict=False)

    assert loaded.model_dir == LONGCAT_INT8_DIR
    assert loaded.quantization is not None
