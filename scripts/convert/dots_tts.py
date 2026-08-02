#!/usr/bin/env python3
"""Convert pinned official dots.tts weights into strict MLX-native base artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from numpy._core.multiarray import _reconstruct as _numpy_reconstruct
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy_safetensors

from mlx_speech.models.dots_tts.checkpoint import (
    BASE_DTYPE_POLICY,
    INT8_DTYPE_POLICY,
    SOURCE_REVISIONS,
    TOKENIZER_FILES,
    DotsTTSCoreComponents,
    DotsTTSQuantizationConfig,
    _strict_load,
    eligible_qwen_quantization_paths,
    quantize_dots_tts_core,
    storage_dtype,
    validate_artifact_dir,
)
from mlx_speech.models.dots_tts.config import DotsTTSConfig


@dataclass(frozen=True)
class FileConversionReport:
    source_file: str
    output_file: str
    source_tensors: int
    output_tensors: int
    folded_weight_norm_pairs: int = 0
    ignored_source_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversionReport:
    variant: str
    artifact_class: str
    source_dir: str
    output_dir: str
    source_manifest_sha256: str
    files: tuple[FileConversionReport, ...]
    quantized_paths: tuple[str, ...] = ()
    source_bytes: int | None = None
    output_bytes: int | None = None


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): __import__("_codecs").encode,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"forbidden latent-statistics pickle global: {module}.{name}"
            ) from error

    def persistent_load(self, persistent_id):
        raise pickle.UnpicklingError(
            f"latent-statistics pickle uses forbidden persistent data: {persistent_id!r}"
        )


def read_latent_statistics(path: str | Path) -> dict[str, np.ndarray]:
    """Read the pinned inline-NumPy torch archive without importing Torch."""

    source = Path(path)
    try:
        with zipfile.ZipFile(source) as archive:
            pickle_names = [
                name for name in archive.namelist() if name.endswith("/data.pkl")
            ]
            if len(pickle_names) != 1:
                raise ValueError("latent statistics archive must contain one data.pkl")
            payload = _RestrictedNumpyUnpickler(
                io.BytesIO(archive.read(pickle_names[0]))
            ).load()
    except (OSError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid latent statistics archive: {source}") from error
    if not isinstance(payload, dict) or set(payload) != {"mean", "var"}:
        raise ValueError("latent statistics must contain exactly mean and var")
    result = {}
    for key in ("mean", "var"):
        value = np.asarray(payload[key])
        if value.dtype != np.float32 or value.shape != (128,):
            raise ValueError(
                f"latent statistics {key} must be float32[128], got {value.dtype}{value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"latent statistics {key} contains non-finite values")
        result[key] = np.ascontiguousarray(value)
    if not np.all(result["var"] > 0):
        raise ValueError("latent statistics variance must be strictly positive")
    return result


def _insert(
    output: dict[str, mx.array],
    key: str,
    value: mx.array,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> None:
    if key in output:
        raise ValueError(f"duplicate converted tensor key: {key}")
    output[key] = value.astype(dtype)


def _core_key(source_key: str) -> str:
    if source_key.startswith("llm.model."):
        return "qwen.model." + source_key.removeprefix("llm.model.")
    if source_key.startswith("eos_proj.0."):
        return "qwen.eos_proj.linear1." + source_key.removeprefix("eos_proj.0.")
    if source_key.startswith("eos_proj.2."):
        return "qwen.eos_proj.linear2." + source_key.removeprefix("eos_proj.2.")
    if source_key.startswith("patch_encoder."):
        return "semantic_encoder." + source_key.removeprefix("patch_encoder.")
    if source_key.startswith("velocity_field_predictor."):
        key = source_key.removeprefix("velocity_field_predictor.")
        key = re.sub(r"^(time|duration)_embedder\.mlp\.0\.", r"\1_embedder.fc1.", key)
        key = re.sub(r"^(time|duration)_embedder\.mlp\.2\.", r"\1_embedder.fc2.", key)
        key = re.sub(
            r"^(blocks\.\d+)\.adaLN_modulation\.1\.",
            r"\1.adaLN_modulation.",
            key,
        )
        key = key.replace(
            "output_layer.adaLN_modulation.1.",
            "output_layer.adaLN_modulation.",
        )
        return "dit." + key
    for source_prefix, target_prefix in (
        ("coordinate_proj.", "coordinate_projection."),
        ("hidden_proj.", "hidden_projection."),
        ("latent_proj.", "latent_projection."),
        ("xvec_proj.0.", "speaker_projection."),
        ("xvec_proj.1.", "speaker_projection_norm."),
    ):
        if source_key.startswith(source_prefix):
            return target_prefix + source_key.removeprefix(source_prefix)
    raise ValueError(f"unmapped core tensor: {source_key}")


def remap_core_weights(
    weights: Mapping[str, mx.array], *, meanflow: bool
) -> dict[str, mx.array]:
    output = {}
    for source_key, value in weights.items():
        target_key = _core_key(source_key)
        if source_key == "patch_encoder.ds_proj.weight":
            value = value.transpose(0, 2, 1)
        _insert(output, target_key, value)
    duration_keys = [key for key in output if key.startswith("dit.duration_embedder.")]
    if meanflow != bool(duration_keys):
        raise ValueError(
            "MeanFlow config and duration-embedding checkpoint tensors disagree"
        )
    return output


def fold_vocoder_weight_norm(
    weights: Mapping[str, mx.array],
) -> tuple[dict[str, mx.array], int]:
    output: dict[str, mx.array] = {}
    handled: set[str] = set()
    folded = 0
    for key, value in weights.items():
        if not key.endswith(".weight_v"):
            continue
        prefix = key.removesuffix(".weight_v")
        g_key = prefix + ".weight_g"
        if g_key not in weights:
            raise ValueError(f"weight_v has no matching weight_g: {key}")
        vector = value.astype(mx.float32)
        scale = weights[g_key].astype(mx.float32)
        axes = tuple(range(1, vector.ndim))
        norm = mx.sqrt(mx.sum(vector * vector, axis=axes, keepdims=True))
        target = prefix + ".weight"
        if target in output:
            raise ValueError(f"duplicate folded vocoder tensor key: {target}")
        output[target] = scale * vector / mx.maximum(norm, 1e-12)
        handled.update((key, g_key))
        folded += 1
    for key, value in weights.items():
        if key in handled:
            continue
        if key.endswith(".weight_g"):
            raise ValueError(f"weight_g has no matching weight_v: {key}")
        if key in output:
            raise ValueError(f"duplicate folded vocoder tensor key: {key}")
        output[key] = value
    return output, folded


def _mi_key(key: str) -> str | None:
    match = re.match(r"^(enc_mi_layer|dec_mi_layer)\.([012])\.(.+)$", key)
    if not match:
        return None
    prefix, index, suffix = match.groups()
    if index == "0":
        return f"{prefix}.input.{suffix}"
    if index == "2":
        return f"{prefix}.output.{suffix}"
    recurrent = re.fullmatch(r"lstm\.(weight|bias)_(ih|hh)_l(\d+)", suffix)
    if recurrent is None:
        raise ValueError(f"unmapped vocoder recurrent tensor: {key}")
    kind, direction, layer = recurrent.groups()
    return f"{prefix}.recurrent.layers.{layer}.{kind}_{direction}"


def _audio_encoder_key(key: str, downsample_stages: int) -> str | None:
    prefix = "audio_encoder.generator."
    if not key.startswith(prefix):
        return None
    suffix = key.removeprefix(prefix)
    match = re.fullmatch(r"(\d+)\.layer\.(weight|bias)", suffix)
    if match:
        index, field = int(match.group(1)), match.group(2)
        if index == 0:
            return f"audio_encoder.pre_conv.{field}"
        if index == 2 + 3 * downsample_stages:
            return f"audio_encoder.post_conv.{field}"
        if index >= 2 and (index - 2) % 3 == 0:
            stage = (index - 2) // 3
            if stage < downsample_stages:
                return f"audio_encoder.down_convs.{stage}.{field}"
    match = re.fullmatch(
        r"(\d+)\.layers\.(\d+)\.([25])\.(weight|bias)", suffix
    )
    if match:
        index, layer, branch, field = match.groups()
        index = int(index)
        if index >= 3 and (index - 3) % 3 == 0:
            stage = (index - 3) // 3
            if stage < downsample_stages:
                target_branch = "convs1" if branch == "2" else "convs2"
                return (
                    f"audio_encoder.residual_stacks.{stage}."
                    f"{target_branch}.{layer}.{field}"
                )
    raise ValueError(f"unmapped AudioVAE encoder tensor: {key}")


def _activation_key(key: str) -> tuple[str, str] | None:
    for suffix, target in (
        (".act.alpha", ".alpha"),
        (".act.beta", ".beta"),
        (".upsample.filter", ".up_filter"),
        (".downsample.lowpass.filter", ".down_filter"),
    ):
        if key.endswith(suffix):
            return key.removesuffix(suffix), target
    return None


def _vocoder_key(key: str, config: DotsTTSConfig) -> tuple[str, str]:
    mi = _mi_key(key)
    if mi is not None:
        return mi, "linear"
    encoder = _audio_encoder_key(key, len(config.vocoder.downsample_rates))
    if encoder is not None:
        return encoder, "conv1d" if key.endswith(".weight") else "other"
    if key.startswith(("pre_proj.", "post_proj.")):
        return key, "conv1d" if key.endswith(".weight") else "other"
    if key.startswith(("decoder.conv_pre.", "decoder.conv_post.")):
        return key, "conv1d" if key.endswith(".weight") else "other"
    upsample = re.fullmatch(r"decoder\.ups\.(\d+)\.0\.(weight|bias)", key)
    if upsample:
        return (
            f"decoder.ups.{upsample.group(1)}.{upsample.group(2)}",
            "conv_transpose" if upsample.group(2) == "weight" else "other",
        )
    residual = re.fullmatch(
        r"decoder\.resblocks\.(\d+)\.(convs[12])\.(\d+)\.(weight|bias)", key
    )
    if residual:
        flat, branch, layer, field = residual.groups()
        kernels = len(config.vocoder.resblock_kernel_sizes)
        stage, kernel = divmod(int(flat), kernels)
        return (
            f"decoder.resblocks.{stage}.{kernel}.{branch}.{layer}.{field}",
            "conv1d" if field == "weight" else "other",
        )
    activation = _activation_key(key)
    if activation is not None:
        source_prefix, target_suffix = activation
        residual_activation = re.fullmatch(
            r"decoder\.resblocks\.(\d+)\.activations\.(\d+)", source_prefix
        )
        if residual_activation:
            flat, activation_index = residual_activation.groups()
            kernels = len(config.vocoder.resblock_kernel_sizes)
            stage, kernel = divmod(int(flat), kernels)
            return (
                f"decoder.resblocks.{stage}.{kernel}.activations."
                f"{activation_index}{target_suffix}",
                "filter" if target_suffix.endswith("filter") else "other",
            )
        if source_prefix == "decoder.activation_post":
            return (
                "decoder.activation_post" + target_suffix,
                "filter" if target_suffix.endswith("filter") else "other",
            )
    raise ValueError(f"unmapped vocoder tensor: {key}")


def remap_vocoder_weights(
    weights: Mapping[str, mx.array], config: DotsTTSConfig
) -> tuple[dict[str, mx.array], int]:
    folded, pair_count = fold_vocoder_weight_norm(weights)
    if pair_count != 80:
        raise ValueError(f"expected 80 vocoder weight-normalization pairs, got {pair_count}")
    output = {}
    for source_key, value in folded.items():
        target_key, kind = _vocoder_key(source_key, config)
        if kind == "conv1d":
            value = value.transpose(0, 2, 1)
        elif kind == "conv_transpose":
            value = value.transpose(1, 2, 0)
        elif kind == "filter":
            source_prefix, _ = _activation_key(source_key)  # type: ignore[misc]
            channels = int(folded[source_prefix + ".act.alpha"].shape[0])
            value = value.transpose(0, 2, 1)
            if int(value.shape[0]) == 1 and channels != 1:
                value = mx.broadcast_to(
                    value, (channels, int(value.shape[1]), 1)
                )
            if int(value.shape[0]) != channels:
                raise ValueError(f"alias-free filter channel mismatch: {source_key}")
        _insert(
            output,
            target_key,
            value,
            dtype=storage_dtype(BASE_DTYPE_POLICY, "vocoder", target_key),
        )
    return output, pair_count


def _speaker_key(key: str) -> str:
    key = key.removeprefix("model.")
    shortcut = re.fullmatch(
        r"(head\.layer[12]\.\d+)\.shortcut\.([01])\.(.+)", key
    )
    if shortcut:
        prefix, index, suffix = shortcut.groups()
        name = "shortcut" if index == "0" else "shortcut_bn"
        return f"{prefix}.{name}.{suffix}"
    if key.startswith("xvector.tdnn."):
        key = "tdnn." + key.removeprefix("xvector.tdnn.")
        return key.replace("nonlinear.batchnorm.", "nonlinear.")
    dense_block = re.fullmatch(
        r"xvector\.block([123])\.tdnnd(\d+)\.(.+)", key
    )
    if dense_block:
        block, layer, suffix = dense_block.groups()
        suffix = suffix.replace("nonlinear1.batchnorm.", "nonlinear1.")
        suffix = suffix.replace("nonlinear2.batchnorm.", "nonlinear2.")
        return f"blocks.{int(block) - 1}.layers.{int(layer) - 1}.{suffix}"
    transit = re.fullmatch(r"xvector\.transit([123])\.(.+)", key)
    if transit:
        index, suffix = transit.groups()
        suffix = suffix.replace("nonlinear.batchnorm.", "nonlinear.")
        return f"transits.{int(index) - 1}.{suffix}"
    if key.startswith("xvector.out_nonlinear.batchnorm."):
        return "out_nonlinear." + key.removeprefix(
            "xvector.out_nonlinear.batchnorm."
        )
    if key.startswith("xvector.dense.linear."):
        return "dense." + key.removeprefix("xvector.dense.linear.")
    if key.startswith("xvector.dense.nonlinear.batchnorm."):
        return "dense_norm." + key.removeprefix(
            "xvector.dense.nonlinear.batchnorm."
        )
    if key.startswith("head."):
        return key
    raise ValueError(f"unmapped speaker tensor: {key}")


def remap_speaker_weights(
    weights: Mapping[str, mx.array],
    *,
    dtype: mx.Dtype = mx.float32,
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    output = {}
    ignored = []
    for source_key, value in weights.items():
        if source_key == "resample.kernel" or source_key.endswith(
            ".num_batches_tracked"
        ):
            ignored.append(source_key)
            continue
        target_key = _speaker_key(source_key)
        if value.ndim == 3:
            value = value.transpose(0, 2, 1)
        elif value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        _insert(output, target_key, value, dtype=dtype)
    if len(ignored) != 123:
        raise ValueError(
            f"expected 122 BatchNorm counters plus resample kernel, got {len(ignored)}"
        )
    return output, tuple(sorted(ignored))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(source: Path, variant: str) -> tuple[Path, str]:
    manifest_path = source.parent.parent / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dots.tts source manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported dots.tts source manifest schema")
    record = payload.get("variants", {}).get(variant)
    expected = SOURCE_REVISIONS[variant]
    if not isinstance(record, dict) or (
        record.get("requested_repo_id") != expected["repo_id"]
        or record.get("resolved_repo_id") != expected["resolved_repo_id"]
        or record.get("revision") != expected["revision"]
    ):
        raise ValueError("source manifest does not match the pinned variant revision")
    consumed_files = (
        "config.json",
        "llm_config.json",
        "model.safetensors",
        "vocoder.safetensors",
        "speaker_encoder.safetensors",
        "latent_stats.pt",
        *sorted(TOKENIZER_FILES),
    )
    files = record.get("files")
    if not isinstance(files, dict):
        raise ValueError("source manifest variant must contain file records")
    for name in consumed_files:
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"source manifest is missing file record: {name}")
        expected_size = entry.get("bytes")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            raise ValueError(f"source manifest file record is incomplete: {name}")
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"dots.tts source file is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"dots.tts source integrity failed for {name}: "
                f"size={actual_size}, expected={expected_size}"
            )
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"dots.tts source integrity failed for {name}: "
                f"sha256={actual_sha}, expected={expected_sha}"
            )
    return manifest_path, _sha256(manifest_path)


def _save(path: Path, weights: dict[str, mx.array]) -> None:
    mx.save_safetensors(str(path), weights)


def _tensor_count(path: Path) -> int:
    with safe_open(path, framework="numpy") as handle:
        return len(handle.keys())


def _artifact_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _validate_source_accounting(report: ConversionReport) -> None:
    for file_report in report.files:
        accounted = (
            file_report.output_tensors
            + file_report.folded_weight_norm_pairs
            + len(file_report.ignored_source_keys)
        )
        if accounted != file_report.source_tensors:
            raise ValueError(
                f"source tensor accounting failed for {file_report.source_file}: "
                f"source={file_report.source_tensors}, accounted={accounted}"
            )


def _stage_report(path: Path, report: ConversionReport) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def convert_variant(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
) -> ConversionReport:
    if variant not in SOURCE_REVISIONS:
        raise ValueError(f"unsupported dots.tts variant: {variant}")
    source = Path(source_dir)
    output = Path(output_dir)
    if output.name != "mlx-base":
        raise ValueError("base dots.tts conversion output must be named mlx-base")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    _manifest_path, manifest_sha = _source_manifest(source, variant)
    config = DotsTTSConfig.from_path(source)
    expected_mode = "meanflow" if variant == "mf" else "flow_matching"
    if config.mode != expected_mode:
        raise ValueError(f"source variant/config mode mismatch: {variant}/{config.mode}")

    statistics = read_latent_statistics(source / "latent_stats.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent)
    )
    report_staging: Path | None = None
    try:
        core_source = mx.load(str(source / "model.safetensors"))
        core = remap_core_weights(core_source, meanflow=config.mode == "meanflow")
        core_source_count = len(core_source)
        core_count = len(core)
        _save(staging / "core.safetensors", core)
        del core_source, core

        vocoder_source = mx.load(str(source / "vocoder.safetensors"))
        vocoder, folded = remap_vocoder_weights(vocoder_source, config)
        vocoder_source_count = len(vocoder_source)
        vocoder_count = len(vocoder)
        _save(staging / "vocoder.safetensors", vocoder)
        del vocoder_source, vocoder

        speaker_source = mx.load(str(source / "speaker_encoder.safetensors"))
        speaker, ignored = remap_speaker_weights(speaker_source)
        speaker_source_count = len(speaker_source)
        speaker_count = len(speaker)
        _save(staging / "speaker.safetensors", speaker)
        del speaker_source, speaker

        save_numpy_safetensors(statistics, staging / "latent_stats.safetensors")
        shutil.copy2(source / "config.json", staging / "config.json")
        shutil.copy2(source / "llm_config.json", staging / "llm_config.json")
        tokenizer = staging / "tokenizer"
        tokenizer.mkdir()
        for name in sorted(TOKENIZER_FILES):
            shutil.copy2(source / name, tokenizer / name)

        pinned = SOURCE_REVISIONS[variant]
        metadata = {
            "schema_version": 1,
            "model_family": "dots_tts",
            "variant": variant,
            "mode": config.mode,
            "artifact_class": "base",
            "source": {
                **pinned,
                "manifest_sha256": manifest_sha,
            },
            "dtype_policy": BASE_DTYPE_POLICY,
            "quantization": None,
        }
        (staging / "mlx_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = ConversionReport(
            variant=variant,
            artifact_class="base",
            source_dir=str(source),
            output_dir=str(output),
            source_manifest_sha256=manifest_sha,
            files=(
                FileConversionReport(
                    "model.safetensors",
                    "core.safetensors",
                    core_source_count,
                    core_count,
                ),
                FileConversionReport(
                    "vocoder.safetensors",
                    "vocoder.safetensors",
                    vocoder_source_count,
                    vocoder_count,
                    folded_weight_norm_pairs=folded,
                ),
                FileConversionReport(
                    "speaker_encoder.safetensors",
                    "speaker.safetensors",
                    speaker_source_count,
                    speaker_count,
                    ignored_source_keys=ignored,
                ),
                FileConversionReport(
                    "latent_stats.pt", "latent_stats.safetensors", 2, 2
                ),
            ),
        )
        _validate_source_accounting(report)
        validate_artifact_dir(staging)
        report_path = output.parent / f"{output.name}-conversion.json"
        report_staging = _stage_report(report_path, report)

        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty artifact: {output}"
                )
            output.rmdir()
        os.replace(staging, output)
        try:
            os.replace(report_staging, report_path)
        except BaseException:
            os.replace(output, staging)
            raise
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if report_staging is not None:
            report_staging.unlink(missing_ok=True)


def quantize_variant(
    base_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
) -> ConversionReport:
    """Build a self-contained selective-int8 artifact from a verified base."""

    if variant not in SOURCE_REVISIONS:
        raise ValueError(f"unsupported dots.tts variant: {variant}")
    base = Path(base_dir)
    output = Path(output_dir)
    if base.name != "mlx-base":
        raise ValueError("dots.tts int8 input must be named mlx-base")
    if output.name != "mlx-int8":
        raise ValueError("int8 dots.tts conversion output must be named mlx-int8")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")

    layout = validate_artifact_dir(base)
    if layout.artifact_config.artifact_class != "base":
        raise ValueError("dots.tts int8 conversion requires a base artifact")
    if layout.artifact_config.variant != variant:
        raise ValueError("base artifact variant does not match requested int8 variant")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent)
    )
    report_staging: Path | None = None
    try:
        core = DotsTTSCoreComponents(layout.config, layout.qwen_config)
        _strict_load(
            "core",
            core,
            base / "core.safetensors",
            expected_dtype=lambda path: storage_dtype(
                layout.artifact_config.dtype_policy,
                "core",
                path,
            ),
        )
        quantized_paths = eligible_qwen_quantization_paths(core, group_size=64)
        quantization = DotsTTSQuantizationConfig(
            bits=8,
            group_size=64,
            mode="affine",
            module_types=("Linear", "Embedding"),
            path_prefixes=("qwen.model.",),
            quantized_paths=quantized_paths,
        )
        quantization.validate()
        quantize_dots_tts_core(core, quantization)
        core_weights = tree_flatten(core.parameters(), destination={})
        mx.eval(list(core_weights.values()))
        _save(staging / "core.safetensors", core_weights)

        copied_files = (
            "config.json",
            "llm_config.json",
            "vocoder.safetensors",
            "speaker.safetensors",
            "latent_stats.safetensors",
        )
        for name in copied_files:
            shutil.copy2(base / name, staging / name)
            if _sha256(base / name) != _sha256(staging / name):
                raise RuntimeError(f"dots.tts int8 copy verification failed: {name}")
        shutil.copytree(base / "tokenizer", staging / "tokenizer")

        metadata = layout.artifact_config.to_dict()
        metadata.update(
            {
                "artifact_class": "int8",
                "dtype_policy": INT8_DTYPE_POLICY,
                "quantization": quantization.to_dict(),
            }
        )
        (staging / "mlx_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_artifact_dir(staging)

        source_bytes = _artifact_bytes(base)
        output_bytes = _artifact_bytes(staging)
        if output_bytes * 4 > source_bytes * 3:
            raise ValueError(
                "dots.tts int8 artifact must be at least 25% smaller than base: "
                f"base_bytes={source_bytes}, int8_bytes={output_bytes}"
            )
        report = ConversionReport(
            variant=variant,
            artifact_class="int8",
            source_dir=str(base),
            output_dir=str(output),
            source_manifest_sha256=layout.artifact_config.source_manifest_sha256,
            files=(
                FileConversionReport(
                    "core.safetensors",
                    "core.safetensors",
                    _tensor_count(base / "core.safetensors"),
                    len(core_weights),
                ),
                *(
                    FileConversionReport(
                        name,
                        name,
                        _tensor_count(base / name),
                        _tensor_count(staging / name),
                    )
                    for name in (
                        "vocoder.safetensors",
                        "speaker.safetensors",
                        "latent_stats.safetensors",
                    )
                ),
            ),
            quantized_paths=quantized_paths,
            source_bytes=source_bytes,
            output_bytes=output_bytes,
        )
        report_path = output.parent / f"{output.name}-conversion.json"
        report_staging = _stage_report(report_path, report)

        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty artifact: {output}"
                )
            output.rmdir()
        os.replace(staging, output)
        try:
            os.replace(report_staging, report_path)
        except BaseException:
            os.replace(output, staging)
            raise
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if report_staging is not None:
            report_staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("soar", "mf", "all"), default="all")
    parser.add_argument("--precision", choices=("base", "int8"), default="base")
    parser.add_argument("--root", type=Path, default=Path("models/dots_tts"))
    args = parser.parse_args()
    variants = ("soar", "mf") if args.variant == "all" else (args.variant,)
    for variant in variants:
        if args.precision == "base":
            report = convert_variant(
                args.root / variant / "original",
                args.root / variant / "mlx-base",
                variant=variant,
            )
        else:
            report = quantize_variant(
                args.root / variant / "mlx-base",
                args.root / variant / "mlx-int8",
                variant=variant,
            )
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
