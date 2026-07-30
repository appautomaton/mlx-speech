from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.dots_tts.checkpoint import load_dots_tts_components


@pytest.mark.parametrize("variant", ("soar", "mf"))
def test_dots_tts_base_strict_loads_every_component(variant: str) -> None:
    artifact = Path("models/dots_tts") / variant / "mlx-base"
    if not artifact.is_dir():
        pytest.skip(f"local dots.tts {variant} base artifact is unavailable")
    loaded = load_dots_tts_components(artifact)
    assert loaded.layout.artifact_config.variant == variant
    assert loaded.layout.artifact_config.artifact_class == "base"
    assert tuple(report.component for report in loaded.reports) == (
        "core",
        "vocoder",
        "speaker",
    )
    assert all(report.is_exact_match for report in loaded.reports)
    assert all(not report.dtype_mismatches for report in loaded.reports)
    assert all(not report.runtime_dtype_mismatches for report in loaded.reports)
    del loaded
    gc.collect()
    mx.clear_cache()
