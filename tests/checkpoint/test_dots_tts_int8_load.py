from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.checkpoint import load_dots_tts_components


@pytest.mark.parametrize("variant", ("soar", "mf"))
def test_dots_tts_int8_strict_loads_exact_selective_predicate(variant: str) -> None:
    artifact = Path("models/dots_tts") / variant / "mlx-int8"
    if not artifact.is_dir():
        pytest.skip(f"local dots.tts {variant} int8 artifact is unavailable")
    loaded = load_dots_tts_components(artifact)
    metadata = loaded.layout.artifact_config
    assert metadata.variant == variant
    assert metadata.artifact_class == "int8"
    assert metadata.quantization is not None
    assert metadata.quantization.path_prefixes == ("qwen.model.",)
    assert metadata.quantization.quantized_paths
    assert all(report.is_exact_match for report in loaded.reports)
    assert all(not report.dtype_mismatches for report in loaded.reports)
    assert all(not report.runtime_dtype_mismatches for report in loaded.reports)

    parameters = tree_flatten(loaded.core.parameters(), destination={})
    for path in metadata.quantization.quantized_paths:
        assert parameters[f"{path}.weight"].dtype == mx.uint32
        assert parameters[f"{path}.scales"].dtype == mx.bfloat16
        assert parameters[f"{path}.biases"].dtype == mx.bfloat16
    assert parameters["qwen.eos_proj.linear1.weight"].dtype == mx.bfloat16
    assert parameters["semantic_encoder.ds_proj.weight"].dtype == mx.bfloat16

    base = artifact.parent / "mlx-base"
    int8_bytes = sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
    base_bytes = sum(path.stat().st_size for path in base.rglob("*") if path.is_file())
    assert int8_bytes * 4 <= base_bytes * 3
    del loaded
    gc.collect()
    mx.clear_cache()
