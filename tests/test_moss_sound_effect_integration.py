from __future__ import annotations

import pytest

from mlx_speech.models.moss_delay import load_moss_sound_effect_model, resolve_moss_sound_effect_model_dir

pytestmark = pytest.mark.local_integration


def test_default_moss_sound_effect_runtime_loads_quantized_mlx_model() -> None:
    resolved = resolve_moss_sound_effect_model_dir()
    loaded = load_moss_sound_effect_model()

    assert resolved.as_posix().endswith("models/openmoss/moss_sound_effect/mlx-4bit")
    assert loaded.alignment_report.is_exact_match
    assert loaded.quantization is not None
