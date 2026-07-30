from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_speech.models.dots_tts.checkpoint import (
    BF16_DTYPE_POLICY,
    INT8_DTYPE_POLICY,
    SOURCE_REVISIONS,
    TOKENIZER_FILES,
    validate_artifact_dir,
)
from test_dots_tts_config import dots_config, qwen_config


def _metadata(*, variant: str = "soar", precision: str = "bf16") -> dict:
    source = SOURCE_REVISIONS[variant]
    quantization = None
    if precision == "int8":
        quantization = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "module_types": ["Linear", "Embedding"],
            "path_prefixes": ["llm."],
            "quantized_paths": ["llm.layers.0.self_attn.q_proj"],
        }
    return {
        "schema_version": 1,
        "model_family": "dots_tts",
        "variant": variant,
        "mode": "meanflow" if variant == "mf" else "flow_matching",
        "precision": precision,
        "source": {
            **source,
            "manifest_sha256": "a" * 64,
        },
        "dtype_policy": (
            INT8_DTYPE_POLICY if precision == "int8" else BF16_DTYPE_POLICY
        ),
        "quantization": quantization,
    }


def _write_artifact(
    root: Path, *, variant: str = "soar", precision: str = "bf16"
) -> Path:
    root.mkdir()
    config = dots_config(meanflow=variant == "mf")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "llm_config.json").write_text(
        json.dumps(qwen_config()), encoding="utf-8"
    )
    (root / "mlx_config.json").write_text(
        json.dumps(_metadata(variant=variant, precision=precision)), encoding="utf-8"
    )
    for name in ("core.safetensors", "vocoder.safetensors", "speaker.safetensors"):
        save_file({"weight": np.ones((2, 2), dtype=np.float32)}, root / name)
    save_file(
        {
            "mean": np.zeros(128, dtype=np.float32),
            "var": np.ones(128, dtype=np.float32),
        },
        root / "latent_stats.safetensors",
    )
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    for name in TOKENIZER_FILES:
        (tokenizer / name).write_text("{}", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("variant", "precision"),
    (("soar", "bf16"), ("mf", "bf16"), ("soar", "int8"), ("mf", "int8")),
)
def test_valid_native_artifact_contract(
    tmp_path: Path, variant: str, precision: str
) -> None:
    artifact = _write_artifact(
        tmp_path / f"{variant}-{precision}", variant=variant, precision=precision
    )
    layout = validate_artifact_dir(artifact)
    assert layout.config.mode == ("meanflow" if variant == "mf" else "flow_matching")
    assert layout.artifact_config.precision == precision
    assert len(layout.weight_files) == 4


def test_contract_rejects_missing_and_unexpected_assets(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    (artifact / "speaker.safetensors").unlink()
    (artifact / "source.bin").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="missing=.*speaker.*unexpected=.*source"):
        validate_artifact_dir(artifact)


def test_contract_rejects_metadata_mode_and_quantization_mismatches(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    metadata = _metadata()
    metadata["mode"] = "meanflow"
    (artifact / "mlx_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="variant/mode mismatch"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(tmp_path / "int8", precision="int8")
    metadata = _metadata(precision="int8")
    metadata["quantization"] = None
    (artifact / "mlx_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="require quantization"):
        validate_artifact_dir(artifact)


def test_contract_rejects_invalid_latent_statistics(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    save_file(
        {"mean": np.zeros(128, dtype=np.float32)},
        artifact / "latent_stats.safetensors",
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_artifact_dir(artifact)
