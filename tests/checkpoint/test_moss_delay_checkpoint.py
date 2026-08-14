from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.moss_delay import load_moss_tts_delay_model

MODEL_DIR = Path("models/openmoss/moss_ttsd/mlx-int8")

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(
        not MODEL_DIR.is_dir(),
        reason="MOSS-TTSD checkpoint is not present",
    ),
]


def test_default_ttsd_runtime_loads_quantized_mlx_model() -> None:
    loaded = load_moss_tts_delay_model()

    assert loaded.alignment_report.is_exact_match
    assert loaded.model.config.n_vq == 16
    assert loaded.model.language_model.config.num_hidden_layers == 36
    assert loaded.quantization is not None
