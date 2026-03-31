from __future__ import annotations

from mlx_voice.models.moss_delay import (
    load_moss_tts_delay_model,
    resolve_moss_tts_delay_model_dir,
)


def test_resolve_moss_tts_delay_model_dir_defaults_to_local_quantized_runtime() -> None:
    resolved = resolve_moss_tts_delay_model_dir()

    assert resolved.as_posix().endswith("models/openmoss/moss_ttsd/mlx-int8")


def test_default_ttsd_runtime_loads_quantized_mlx_model() -> None:
    loaded = load_moss_tts_delay_model()

    assert loaded.alignment_report.is_exact_match
    assert loaded.model.config.n_vq == 16
    assert loaded.model.language_model.config.num_hidden_layers == 36
    assert loaded.quantization is not None
