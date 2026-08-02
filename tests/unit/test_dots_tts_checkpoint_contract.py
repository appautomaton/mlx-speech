from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_speech.models.dots_tts.checkpoint import (
    BASE_DTYPE_POLICY,
    INT8_DTYPE_POLICY,
    SOURCE_REVISIONS,
    TOKENIZER_FILES,
    storage_dtype,
    validate_artifact_dir,
)
from test_dots_tts_config import dots_config, qwen_config


def _metadata(*, variant: str = "soar", artifact_class: str = "base") -> dict:
    source = SOURCE_REVISIONS[variant]
    quantization = None
    if artifact_class == "int8":
        quantization = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "module_types": ["Linear", "Embedding"],
            "path_prefixes": ["qwen.model."],
            "quantized_paths": ["qwen.model.embed_tokens"],
        }
    return {
        "schema_version": 1,
        "model_family": "dots_tts",
        "variant": variant,
        "mode": "meanflow" if variant == "mf" else "flow_matching",
        "artifact_class": artifact_class,
        "source": {
            **source,
            "manifest_sha256": "a" * 64,
        },
        "dtype_policy": (
            INT8_DTYPE_POLICY
            if artifact_class == "int8"
            else BASE_DTYPE_POLICY
        ),
        "quantization": quantization,
    }


def _write_artifact(
    root: Path, *, variant: str = "soar", artifact_class: str = "base"
) -> Path:
    root.mkdir()
    config = dots_config(meanflow=variant == "mf")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "llm_config.json").write_text(
        json.dumps(qwen_config()), encoding="utf-8"
    )
    (root / "mlx_config.json").write_text(
        json.dumps(_metadata(variant=variant, artifact_class=artifact_class)),
        encoding="utf-8",
    )
    if artifact_class == "base":
        mx.save_safetensors(
            str(root / "core.safetensors"),
            {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
        )
    else:
        mx.save_safetensors(
            str(root / "core.safetensors"),
            {
                "qwen.model.embed_tokens.weight": mx.ones(
                    (2, 1), dtype=mx.uint32
                ),
                "qwen.model.embed_tokens.scales": mx.ones(
                    (2, 1), dtype=mx.bfloat16
                ),
                "qwen.model.embed_tokens.biases": mx.zeros(
                    (2, 1), dtype=mx.bfloat16
                ),
            },
        )
    mx.save_safetensors(
        str(root / "vocoder.safetensors"),
        {
            "audio_encoder.weight": mx.ones((1,), dtype=mx.float32),
            "enc_mi_layer.weight": mx.ones((1,), dtype=mx.float32),
            "pre_proj.weight": mx.ones((1,), dtype=mx.float32),
            "post_proj.weight": mx.ones((1,), dtype=mx.bfloat16),
            "dec_mi_layer.weight": mx.ones((1,), dtype=mx.bfloat16),
            "decoder.weight": mx.ones((1,), dtype=mx.bfloat16),
        },
    )
    save_file(
        {"weight": np.ones((2, 2), dtype=np.float32)},
        root / "speaker.safetensors",
    )
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
    ("variant", "artifact_class"),
    (("soar", "base"), ("mf", "base"), ("soar", "int8"), ("mf", "int8")),
)
def test_valid_native_artifact_contract(
    tmp_path: Path, variant: str, artifact_class: str
) -> None:
    artifact = _write_artifact(
        tmp_path / f"{variant}-{artifact_class}",
        variant=variant,
        artifact_class=artifact_class,
    )
    layout = validate_artifact_dir(artifact)
    assert layout.config.mode == ("meanflow" if variant == "mf" else "flow_matching")
    assert layout.artifact_config.artifact_class == artifact_class
    assert len(layout.weight_files) == 4


def test_contract_rejects_missing_and_unexpected_assets(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    (artifact / "speaker.safetensors").unlink()
    (artifact / "source.bin").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="missing=.*speaker.*unexpected=.*source"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(tmp_path / "extra-directory")
    (artifact / "source-layout").mkdir()
    with pytest.raises(ValueError, match="directories mismatch"):
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

    artifact = _write_artifact(tmp_path / "int8", artifact_class="int8")
    metadata = _metadata(artifact_class="int8")
    metadata["quantization"] = None
    (artifact / "mlx_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="require quantization"):
        validate_artifact_dir(artifact)


def test_int8_contract_rejects_incomplete_quantization_and_base_dtype_drift(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "incomplete-int8",
        artifact_class="int8",
    )
    core = mx.load(str(artifact / "core.safetensors"))
    del core["qwen.model.embed_tokens.scales"]
    mx.save_safetensors(str(artifact / "core.safetensors"), core)
    with pytest.raises(ValueError, match="missing quantized tensors"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(
        tmp_path / "wrong-int8-base-dtype",
        artifact_class="int8",
    )
    vocoder = mx.load(str(artifact / "vocoder.safetensors"))
    vocoder["audio_encoder.weight"] = vocoder["audio_encoder.weight"].astype(
        mx.bfloat16
    )
    mx.save_safetensors(str(artifact / "vocoder.safetensors"), vocoder)
    with pytest.raises(ValueError, match="must be float32"):
        validate_artifact_dir(artifact)


def test_contract_rejects_invalid_latent_statistics(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    save_file(
        {"mean": np.zeros(128, dtype=np.float32)},
        artifact / "latent_stats.safetensors",
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(tmp_path / "wrong-latent-shape")
    save_file(
        {
            "mean": np.zeros(127, dtype=np.float32),
            "var": np.ones(127, dtype=np.float32),
        },
        artifact / "latent_stats.safetensors",
    )
    with pytest.raises(ValueError, match=r"must have shape \(128,\)"):
        validate_artifact_dir(artifact)


def test_contract_rejects_wrong_and_uncovered_base_dtypes(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "wrong-dtype")
    save_file(
        {"weight": np.ones((2, 2), dtype=np.float32)},
        artifact / "core.safetensors",
    )
    with pytest.raises(ValueError, match="must be bfloat16"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(tmp_path / "uncovered")
    mx.save_safetensors(
        str(artifact / "vocoder.safetensors"),
        {"unknown.weight": mx.ones((1,), dtype=mx.bfloat16)},
    )
    with pytest.raises(ValueError, match="exactly once"):
        validate_artifact_dir(artifact)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("vocoder-encoder", "must be float32"),
        ("vocoder-decoder", "must be bfloat16"),
        ("speaker", "must be float32"),
        ("latent", "must be float32"),
    ),
)
def test_contract_rejects_each_mixed_policy_violation(
    tmp_path: Path, case: str, expected: str
) -> None:
    artifact = _write_artifact(tmp_path / case)
    if case.startswith("vocoder"):
        weights = mx.load(str(artifact / "vocoder.safetensors"))
        key = (
            "audio_encoder.weight"
            if case == "vocoder-encoder"
            else "decoder.weight"
        )
        wrong_dtype = mx.bfloat16 if case == "vocoder-encoder" else mx.float32
        weights[key] = weights[key].astype(wrong_dtype)
        mx.save_safetensors(str(artifact / "vocoder.safetensors"), weights)
    elif case == "speaker":
        mx.save_safetensors(
            str(artifact / "speaker.safetensors"),
            {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
        )
    else:
        save_file(
            {
                "mean": np.zeros(128, dtype=np.float64),
                "var": np.ones(128, dtype=np.float32),
            },
            artifact / "latent_stats.safetensors",
        )
    with pytest.raises(ValueError, match=expected):
        validate_artifact_dir(artifact)


def test_contract_rejects_obsolete_bf16_directory_name(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "mlx-bf16")
    with pytest.raises(ValueError, match="obsolete mlx-bf16"):
        validate_artifact_dir(artifact)

    artifact = _write_artifact(tmp_path / "mlx-int8")
    with pytest.raises(ValueError, match="class base requires directory mlx-base"):
        validate_artifact_dir(artifact)


@pytest.mark.parametrize("dtype_policy", (BASE_DTYPE_POLICY, INT8_DTYPE_POLICY))
def test_base_and_int8_vocoder_decoder_execution_paths_are_bf16(
    dtype_policy: dict[str, dict[str, str]],
) -> None:
    for path in (
        "post_proj.weight",
        "dec_mi_layer.input.weight",
        "dec_mi_layer.recurrent.layers.0.weight_ih",
        "decoder.conv_pre.weight",
    ):
        assert storage_dtype(dtype_policy, "vocoder", path) == mx.bfloat16
