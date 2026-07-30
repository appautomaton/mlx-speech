from __future__ import annotations

import hashlib
import json
import os
import pickle
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.config import DotsTTSConfig
from mlx_speech.models.dots_tts.checkpoint import SOURCE_REVISIONS, TOKENIZER_FILES
from scripts.convert import dots_tts as converter
from scripts.convert.dots_tts import (
    _speaker_key,
    _vocoder_key,
    fold_vocoder_weight_norm,
    read_latent_statistics,
    remap_core_weights,
    remap_speaker_weights,
)
from test_dots_tts_config import dots_config, qwen_config


def test_core_remap_uses_runtime_names_and_native_conv_layout() -> None:
    source = {
        "llm.model.embed_tokens.weight": mx.zeros((7, 4)),
        "eos_proj.0.weight": mx.zeros((4, 4)),
        "patch_encoder.ds_proj.weight": mx.zeros((3, 2, 5)),
        "velocity_field_predictor.time_embedder.mlp.0.weight": mx.zeros((8, 256)),
        "velocity_field_predictor.blocks.0.adaLN_modulation.1.bias": mx.zeros((48,)),
        "coordinate_proj.weight": mx.zeros((8, 3)),
    }
    converted = remap_core_weights(source, meanflow=False)
    assert set(converted) == {
        "qwen.model.embed_tokens.weight",
        "qwen.eos_proj.linear1.weight",
        "semantic_encoder.ds_proj.weight",
        "dit.time_embedder.fc1.weight",
        "dit.blocks.0.adaLN_modulation.bias",
        "coordinate_projection.weight",
    }
    assert converted["semantic_encoder.ds_proj.weight"].shape == (3, 5, 2)
    assert all(value.dtype == mx.bfloat16 for value in converted.values())


def test_core_remap_requires_duration_weights_only_for_meanflow() -> None:
    duration = {
        "velocity_field_predictor.duration_embedder.mlp.0.weight": mx.zeros((8, 256))
    }
    assert "dit.duration_embedder.fc1.weight" in remap_core_weights(
        duration, meanflow=True
    )
    with pytest.raises(ValueError, match="duration"):
        remap_core_weights(duration, meanflow=False)
    with pytest.raises(ValueError, match="duration"):
        remap_core_weights({}, meanflow=True)


def test_weight_norm_is_folded_once_without_torch() -> None:
    source = {
        "layer.weight_v": mx.array([[[3.0, 4.0]]]),
        "layer.weight_g": mx.array([[[10.0]]]),
        "layer.bias": mx.array([1.0]),
    }
    converted, count = fold_vocoder_weight_norm(source)
    mx.eval(converted)
    assert count == 1
    assert set(converted) == {"layer.weight", "layer.bias"}
    np.testing.assert_allclose(converted["layer.weight"], [[[6.0, 8.0]]])


def test_weight_norm_rejects_a_duplicate_folded_target() -> None:
    source = {
        "layer.weight_v": mx.ones((1, 1, 1)),
        "layer.weight_g": mx.ones((1, 1, 1)),
        "layer.weight": mx.ones((1, 1, 1)),
    }
    with pytest.raises(ValueError, match="duplicate folded"):
        fold_vocoder_weight_norm(source)


def test_vocoder_and_speaker_key_maps_are_explicit() -> None:
    config = DotsTTSConfig.from_dict(dots_config())
    assert _vocoder_key("audio_encoder.generator.3.layers.2.5.weight", config) == (
        "audio_encoder.residual_stacks.0.convs2.2.weight",
        "conv1d",
    )
    assert _vocoder_key("decoder.ups.2.0.weight", config) == (
        "decoder.ups.2.weight",
        "conv_transpose",
    )
    assert _speaker_key(
        "model.xvector.block2.tdnnd7.nonlinear1.batchnorm.running_mean"
    ) == "blocks.1.layers.6.nonlinear1.running_mean"
    assert _speaker_key("model.head.layer1.0.shortcut.1.running_var") == (
        "head.layer1.0.shortcut_bn.running_var"
    )


def test_speaker_remap_preserves_fp32_storage() -> None:
    source = {
        "model.xvector.out_nonlinear.batchnorm.running_mean": mx.ones((3,)),
        **{
            f"training.{index}.num_batches_tracked": mx.array(0)
            for index in range(122)
        },
        "resample.kernel": mx.ones((1,)),
    }
    converted, ignored = remap_speaker_weights(source)
    assert converted["out_nonlinear.running_mean"].dtype == mx.float32
    assert len(ignored) == 123
    assert "resample.kernel" in ignored


def _write_stats_archive(path: Path, payload: object) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))


def test_restricted_latent_reader_accepts_only_expected_numpy_arrays(
    tmp_path: Path,
) -> None:
    source = tmp_path / "latent_stats.pt"
    _write_stats_archive(
        source,
        {
            "mean": np.zeros(128, dtype=np.float32),
            "var": np.ones(128, dtype=np.float32),
        },
    )
    result = read_latent_statistics(source)
    assert set(result) == {"mean", "var"}
    assert result["mean"].dtype == np.float32

    forbidden = tmp_path / "forbidden.pt"
    _write_stats_archive(forbidden, {"mean": eval, "var": np.ones(128, np.float32)})
    with pytest.raises(ValueError, match="invalid latent statistics"):
        read_latent_statistics(forbidden)


def test_restricted_latent_reader_rejects_wrong_shape(tmp_path: Path) -> None:
    source = tmp_path / "latent_stats.pt"
    _write_stats_archive(
        source,
        {
            "mean": np.zeros(127, dtype=np.float32),
            "var": np.ones(127, dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match=r"float32\[128\]"):
        read_latent_statistics(source)


def _fake_conversion_source(tmp_path: Path) -> Path:
    family = tmp_path / "dots_tts"
    source = family / "soar" / "original"
    source.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps(dots_config()), encoding="utf-8")
    (source / "llm_config.json").write_text(
        json.dumps(qwen_config()), encoding="utf-8"
    )
    for name in (
        "model.safetensors",
        "vocoder.safetensors",
        "speaker_encoder.safetensors",
        "latent_stats.pt",
        *TOKENIZER_FILES,
    ):
        (source / name).write_bytes(b"fixture")
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in source.iterdir()
        if path.is_file()
    }
    (family / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variants": {
                    "soar": {
                        "requested_repo_id": SOURCE_REVISIONS["soar"]["repo_id"],
                        "resolved_repo_id": SOURCE_REVISIONS["soar"][
                            "resolved_repo_id"
                        ],
                        "revision": SOURCE_REVISIONS["soar"]["revision"],
                        "files": files,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def _mock_conversion_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(path: str) -> dict[str, mx.array]:
        if path.endswith("model.safetensors"):
            return {"source": mx.ones((1,))}
        if path.endswith("vocoder.safetensors"):
            return {f"source.{index}": mx.ones((1,)) for index in range(81)}
        return {f"source.{index}": mx.ones((1,)) for index in range(124)}

    monkeypatch.setattr(converter.mx, "load", fake_load)
    monkeypatch.setattr(
        converter,
        "remap_core_weights",
        lambda weights, *, meanflow: {"weight": mx.ones((1,), dtype=mx.bfloat16)},
    )
    monkeypatch.setattr(
        converter,
        "remap_vocoder_weights",
        lambda weights, config: ({"decoder.weight": mx.ones((1,))}, 80),
    )
    monkeypatch.setattr(
        converter,
        "remap_speaker_weights",
        lambda weights: (
            {"weight": mx.ones((1,), dtype=mx.float32)},
            tuple(f"ignored.{index}" for index in range(123)),
        ),
    )
    monkeypatch.setattr(
        converter,
        "read_latent_statistics",
        lambda path: {
            "mean": np.zeros(128, dtype=np.float32),
            "var": np.ones(128, dtype=np.float32),
        },
    )


def test_source_manifest_requires_complete_verified_file_records(
    tmp_path: Path,
) -> None:
    source = _fake_conversion_source(tmp_path)
    manifest_path = source.parent.parent / "source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["variants"]["soar"]["files"]["model.safetensors"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing file record: model.safetensors"):
        converter._source_manifest(source, "soar")

    source = _fake_conversion_source(tmp_path / "incomplete")
    manifest_path = source.parent.parent / "source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["variants"]["soar"]["files"]["config.json"]["sha256"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record is incomplete: config.json"):
        converter._source_manifest(source, "soar")


@pytest.mark.parametrize(("field", "value"), (("bytes", 1), ("sha256", "0" * 64)))
def test_source_manifest_rejects_size_and_hash_mismatches(
    tmp_path: Path, field: str, value: int | str
) -> None:
    source = _fake_conversion_source(tmp_path)
    manifest_path = source.parent.parent / "source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["variants"]["soar"]["files"]["config.json"][field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source integrity failed for config.json"):
        converter._source_manifest(source, "soar")


def test_tampered_latent_is_rejected_before_restricted_unpickling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_conversion_source(tmp_path)
    (source / "latent_stats.pt").write_bytes(b"mutated")
    reader_called = False
    model_load_called = False

    def forbidden_reader(path: Path) -> dict[str, np.ndarray]:
        nonlocal reader_called
        reader_called = True
        raise AssertionError("integrity must run before latent reader")

    def forbidden_model_load(path: str) -> dict[str, mx.array]:
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("integrity must run before model loading")

    monkeypatch.setattr(converter, "read_latent_statistics", forbidden_reader)
    monkeypatch.setattr(converter.mx, "load", forbidden_model_load)
    with pytest.raises(ValueError, match="integrity failed for latent_stats.pt"):
        converter.convert_variant(source, source.parent / "mlx-base", variant="soar")
    assert not reader_called
    assert not model_load_called


def test_failed_staging_validation_never_publishes_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_conversion_source(tmp_path)
    output = source.parent / "mlx-base"
    _mock_conversion_payloads(monkeypatch)
    staged = []

    def reject_staging(path: Path) -> None:
        staged.append(Path(path))
        raise ValueError("staged artifact rejected")

    monkeypatch.setattr(converter, "validate_artifact_dir", reject_staging)
    with pytest.raises(ValueError, match="staged artifact rejected"):
        converter.convert_variant(source, output, variant="soar")
    assert len(staged) == 1
    assert staged[0].name.startswith(".mlx-base-staging-")
    assert not output.exists()
    assert not list(output.parent.glob(".mlx-base-staging-*"))


def test_report_promotion_failure_rolls_back_new_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_conversion_source(tmp_path)
    output = source.parent / "mlx-base"
    report_path = source.parent / "mlx-base-conversion.json"
    report_path.write_text("previous report\n", encoding="utf-8")
    _mock_conversion_payloads(monkeypatch)
    monkeypatch.setattr(converter, "validate_artifact_dir", lambda path: None)
    real_replace = os.replace

    def fail_report_promotion(source_path: str | Path, target_path: str | Path) -> None:
        if Path(target_path) == report_path:
            raise OSError("report promotion failed")
        real_replace(source_path, target_path)

    monkeypatch.setattr(converter.os, "replace", fail_report_promotion)
    with pytest.raises(OSError, match="report promotion failed"):
        converter.convert_variant(source, output, variant="soar")
    assert not output.exists()
    assert report_path.read_text(encoding="utf-8") == "previous report\n"
    assert not list(output.parent.glob(".mlx-base-staging-*"))
    assert not list(output.parent.glob(".mlx-base-conversion.json.*.tmp"))


def test_converter_never_overwrites_nonempty_final_artifact(tmp_path: Path) -> None:
    output = tmp_path / "mlx-base"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing artifact"):
        converter.convert_variant(tmp_path / "source", output, variant="soar")
    assert marker.read_text(encoding="utf-8") == "owned"


def test_int8_converter_never_overwrites_nonempty_final_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mlx-int8"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing artifact"):
        converter.quantize_variant(
            tmp_path / "mlx-base",
            output,
            variant="soar",
        )
    assert marker.read_text(encoding="utf-8") == "owned"
