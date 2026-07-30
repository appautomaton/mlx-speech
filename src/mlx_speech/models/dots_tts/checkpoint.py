"""Strict native artifact contract for converted dots.tts checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

from .config import DotsTTSConfig, DotsTTSQwenConfig


ARTIFACT_FILES = {
    "config.json",
    "llm_config.json",
    "mlx_config.json",
    "core.safetensors",
    "vocoder.safetensors",
    "speaker.safetensors",
    "latent_stats.safetensors",
}
TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
SOURCE_REVISIONS = {
    "soar": {
        "repo_id": "rednote-hilab/dots.tts-soar",
        "resolved_repo_id": "dots-studio/dots.tts-soar",
        "revision": "e3520f75254d0020a0406db31c51a79d00d22d55",
    },
    "mf": {
        "repo_id": "rednote-hilab/dots.tts-mf",
        "resolved_repo_id": "dots-studio/dots.tts-mf",
        "revision": "25c53fb462e57087e52237daa5ea30df1c5cc328",
    },
}
BASE_DTYPE_POLICY = {
    "core": {"*": "bfloat16"},
    "vocoder": {
        "audio_encoder.*": "float32",
        "enc_mi_layer.*": "float32",
        "pre_proj.*": "float32",
        "post_proj.*": "bfloat16",
        "dec_mi_layer.*": "bfloat16",
        "decoder.*": "bfloat16",
    },
    "speaker": {"*": "float32"},
    "latent_stats": {"*": "float32"},
}
# Quantization has a separate exact-path predicate in ``quantization``. All
# tensors outside that predicate retain the base policy.
INT8_DTYPE_POLICY = BASE_DTYPE_POLICY

_MLX_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}
_SAFETENSORS_DTYPES = {
    "bfloat16": "BF16",
    "float32": "F32",
    "uint32": "U32",
}


def storage_dtype_name(
    dtype_policy: Mapping[str, Mapping[str, str]], component: str, path: str
) -> str:
    """Resolve exactly one total native-path dtype rule."""

    rules = dtype_policy.get(component)
    if rules is None:
        raise ValueError(f"dots.tts dtype policy has no {component} component")
    matches = [dtype for pattern, dtype in rules.items() if fnmatchcase(path, pattern)]
    if len(matches) != 1:
        raise ValueError(
            f"dots.tts dtype policy must match {component}.{path} exactly once, "
            f"got {len(matches)} matches"
        )
    dtype_name = matches[0]
    if dtype_name not in _MLX_DTYPES:
        raise ValueError(f"unsupported dots.tts storage dtype: {dtype_name}")
    return dtype_name


def storage_dtype(
    dtype_policy: Mapping[str, Mapping[str, str]], component: str, path: str
) -> mx.Dtype:
    return _MLX_DTYPES[storage_dtype_name(dtype_policy, component, path)]


@dataclass(frozen=True)
class DotsTTSQuantizationConfig:
    bits: int
    group_size: int
    mode: str
    module_types: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    quantized_paths: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSQuantizationConfig":
        config = cls(
            bits=int(payload.get("bits", 0)),
            group_size=int(payload.get("group_size", 0)),
            mode=str(payload.get("mode", "")),
            module_types=tuple(
                str(value) for value in payload.get("module_types", ())
            ),
            path_prefixes=tuple(
                str(value) for value in payload.get("path_prefixes", ())
            ),
            quantized_paths=tuple(
                str(value) for value in payload.get("quantized_paths", ())
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.bits != 8 or self.group_size != 64 or self.mode != "affine":
            raise ValueError("dots.tts int8 requires affine 8-bit groups of 64")
        if self.module_types != ("Linear", "Embedding"):
            raise ValueError("dots.tts int8 may target only Linear and Embedding modules")
        if self.path_prefixes != ("qwen.model.",):
            raise ValueError("dots.tts int8 may target only the native qwen.model.* trunk")
        if not self.quantized_paths:
            raise ValueError("dots.tts int8 metadata must name every quantized path")
        if any(
            not path.startswith("qwen.model.") or path.endswith(".")
            for path in self.quantized_paths
        ):
            raise ValueError("dots.tts quantized paths must stay under qwen.model.*")
        if tuple(sorted(set(self.quantized_paths))) != self.quantized_paths:
            raise ValueError("dots.tts quantized paths must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "module_types": list(self.module_types),
            "path_prefixes": list(self.path_prefixes),
            "quantized_paths": list(self.quantized_paths),
        }


def artifact_tensor_dtype_name(
    artifact_config: "DotsTTSArtifactConfig",
    component: str,
    path: str,
) -> str:
    """Resolve stored dtype, including exact affine-quantized tensor fields."""

    quantization = artifact_config.quantization
    if component == "core" and quantization is not None:
        for module_path in quantization.quantized_paths:
            prefix = f"{module_path}."
            if not path.startswith(prefix):
                continue
            field = path.removeprefix(prefix)
            if field == "weight":
                return "uint32"
            if field in {"scales", "biases"}:
                return storage_dtype_name(
                    artifact_config.dtype_policy,
                    component,
                    f"{module_path}.weight",
                )
    return storage_dtype_name(artifact_config.dtype_policy, component, path)


def artifact_tensor_dtype(
    artifact_config: "DotsTTSArtifactConfig",
    component: str,
    path: str,
) -> mx.Dtype:
    dtype_name = artifact_tensor_dtype_name(artifact_config, component, path)
    if dtype_name == "uint32":
        return mx.uint32
    return _MLX_DTYPES[dtype_name]


@dataclass(frozen=True)
class DotsTTSArtifactConfig:
    schema_version: int
    model_family: str
    variant: Literal["soar", "mf"]
    mode: Literal["flow_matching", "meanflow"]
    artifact_class: Literal["base", "int8"]
    source_repo_id: str
    source_resolved_repo_id: str
    source_revision: str
    source_manifest_sha256: str
    dtype_policy: dict[str, dict[str, str]]
    quantization: DotsTTSQuantizationConfig | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSArtifactConfig":
        required = {
            "schema_version",
            "model_family",
            "variant",
            "mode",
            "artifact_class",
            "source",
            "dtype_policy",
            "quantization",
        }
        missing = sorted(required - payload.keys())
        unexpected = sorted(payload.keys() - required)
        if missing or unexpected:
            raise ValueError(
                f"mlx_config.json fields mismatch: missing={missing}, unexpected={unexpected}"
            )
        source = payload["source"]
        if not isinstance(source, dict):
            raise TypeError("mlx_config.json source must be an object")
        config = cls(
            schema_version=int(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            variant=str(payload["variant"]),  # type: ignore[arg-type]
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            artifact_class=str(payload["artifact_class"]),  # type: ignore[arg-type]
            source_repo_id=str(source.get("repo_id", "")),
            source_resolved_repo_id=str(source.get("resolved_repo_id", "")),
            source_revision=str(source.get("revision", "")),
            source_manifest_sha256=str(source.get("manifest_sha256", "")),
            dtype_policy={
                str(component): {
                    str(pattern): str(dtype) for pattern, dtype in rules.items()
                }
                for component, rules in payload["dtype_policy"].items()
            },
            quantization=(
                DotsTTSQuantizationConfig.from_dict(payload["quantization"])
                if payload["quantization"] is not None
                else None
            ),
        )
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "DotsTTSArtifactConfig":
        source = Path(path)
        if source.is_dir():
            source = source / "mlx_config.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.schema_version != 1 or self.model_family != "dots_tts":
            raise ValueError("unsupported dots.tts artifact schema")
        if self.variant not in SOURCE_REVISIONS:
            raise ValueError(f"unsupported dots.tts variant: {self.variant}")
        expected_mode = "meanflow" if self.variant == "mf" else "flow_matching"
        if self.mode != expected_mode:
            raise ValueError(
                f"dots.tts variant/mode mismatch: {self.variant}/{self.mode}"
            )
        expected_source = SOURCE_REVISIONS[self.variant]
        actual_source = {
            "repo_id": self.source_repo_id,
            "resolved_repo_id": self.source_resolved_repo_id,
            "revision": self.source_revision,
        }
        if actual_source != expected_source:
            raise ValueError("dots.tts source provenance does not match pinned revision")
        if len(self.source_manifest_sha256) != 64:
            raise ValueError("dots.tts source manifest SHA-256 is invalid")
        if self.artifact_class not in {"base", "int8"}:
            raise ValueError(
                f"unsupported dots.tts artifact class: {self.artifact_class}"
            )
        expected_policy = (
            BASE_DTYPE_POLICY
            if self.artifact_class == "base"
            else INT8_DTYPE_POLICY
        )
        if self.dtype_policy != expected_policy:
            raise ValueError("dots.tts dtype policy is inconsistent with artifact class")
        if self.artifact_class == "base" and self.quantization is not None:
            raise ValueError("base dots.tts artifacts cannot carry quantization metadata")
        if self.artifact_class == "int8" and self.quantization is None:
            raise ValueError("int8 dots.tts artifacts require quantization metadata")
        if self.quantization is not None:
            self.quantization.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "variant": self.variant,
            "mode": self.mode,
            "artifact_class": self.artifact_class,
            "source": {
                "repo_id": self.source_repo_id,
                "resolved_repo_id": self.source_resolved_repo_id,
                "revision": self.source_revision,
                "manifest_sha256": self.source_manifest_sha256,
            },
            "dtype_policy": self.dtype_policy,
            "quantization": (
                self.quantization.to_dict() if self.quantization is not None else None
            ),
        }


@dataclass(frozen=True)
class DotsTTSArtifactLayout:
    model_dir: Path
    config: DotsTTSConfig
    qwen_config: DotsTTSQwenConfig
    artifact_config: DotsTTSArtifactConfig
    weight_files: tuple[Path, ...]
    tokenizer_dir: Path


class _SupportsWeights(Protocol):
    def parameters(self): ...
    def load_weights(self, file_or_weights, strict: bool = True): ...


@dataclass(frozen=True)
class DotsTTSAlignmentReport:
    component: str
    duplicate_checkpoint_keys: tuple[str, ...]
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    dtype_mismatches: tuple[tuple[str, str, str], ...]
    runtime_dtype_mismatches: tuple[tuple[str, str, str], ...] = ()

    @property
    def is_exact_match(self) -> bool:
        return not (
            self.duplicate_checkpoint_keys
            or self.missing_in_model
            or self.missing_in_checkpoint
            or self.shape_mismatches
            or self.dtype_mismatches
            or self.runtime_dtype_mismatches
        )

    def require_exact(self) -> None:
        if not self.is_exact_match:
            raise ValueError(
                f"dots.tts {self.component} alignment failed: "
                f"duplicate_checkpoint_keys={self.duplicate_checkpoint_keys}, "
                f"missing_in_model={self.missing_in_model}, "
                f"missing_in_checkpoint={self.missing_in_checkpoint}, "
                f"shape_mismatches={self.shape_mismatches}, "
                f"dtype_mismatches={self.dtype_mismatches}, "
                f"runtime_dtype_mismatches={self.runtime_dtype_mismatches}"
            )


class DotsTTSCoreComponents(nn.Module):
    """Weight-bearing core modules stored together in core.safetensors."""

    def __init__(self, config: DotsTTSConfig, qwen_config: DotsTTSQwenConfig):
        super().__init__()
        from .dit import DiT
        from .qwen import DotsTTSQwen
        from .semantic_encoder import VAESemanticEncoder

        hidden_size = config.dit.hidden_size
        self.qwen = DotsTTSQwen(qwen_config)
        self.semantic_encoder = VAESemanticEncoder.from_config(
            config, output_dim=qwen_config.hidden_size
        )
        self.dit = DiT(
            hidden_size,
            config.latent_dim,
            config.dit,
            meanflow=config.mode == "meanflow",
        )
        self.coordinate_projection = nn.Linear(
            config.latent_dim, hidden_size, bias=True
        )
        self.hidden_projection = nn.Linear(
            qwen_config.hidden_size, hidden_size, bias=True
        )
        self.latent_projection = nn.Linear(
            config.latent_dim, hidden_size, bias=True
        )
        self.speaker_projection = nn.Linear(
            config.campplus_embedding_size, hidden_size, bias=True
        )
        self.speaker_projection_norm = nn.LayerNorm(hidden_size)


def eligible_qwen_quantization_paths(
    core: DotsTTSCoreComponents,
    *,
    group_size: int = 64,
) -> tuple[str, ...]:
    """Return the complete native Qwen Linear/Embedding quantization predicate."""

    paths = []
    for path, module in core.named_modules():
        if not path.startswith("qwen.model."):
            continue
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if int(module.weight.shape[-1]) % group_size == 0:
            paths.append(path)
    return tuple(sorted(paths))


def quantize_dots_tts_core(
    core: DotsTTSCoreComponents,
    quantization: DotsTTSQuantizationConfig,
) -> DotsTTSCoreComponents:
    """Reconstruct and apply the exact serialized Qwen-only predicate."""

    quantization.validate()
    eligible = eligible_qwen_quantization_paths(
        core,
        group_size=quantization.group_size,
    )
    if quantization.quantized_paths != eligible:
        missing = tuple(sorted(set(eligible) - set(quantization.quantized_paths)))
        unexpected = tuple(sorted(set(quantization.quantized_paths) - set(eligible)))
        raise ValueError(
            "dots.tts quantized path predicate differs from eligible native Qwen "
            f"modules: missing={missing}, unexpected={unexpected}"
        )
    selected = set(quantization.quantized_paths)
    nn.quantize(
        core,
        group_size=quantization.group_size,
        bits=quantization.bits,
        mode=quantization.mode,
        class_predicate=lambda path, module: path in selected,
    )
    return core


@dataclass(frozen=True)
class LoadedDotsTTSComponents:
    layout: DotsTTSArtifactLayout
    core: DotsTTSCoreComponents
    audio_vae: Any
    speaker_encoder: Any
    latent_io: Any
    reports: tuple[DotsTTSAlignmentReport, ...]


def align_state_dict(
    component: str,
    model: _SupportsWeights,
    weights: Mapping[str, mx.array] | Iterable[tuple[str, mx.array]],
    *,
    expected_dtype: mx.Dtype | Callable[[str], mx.Dtype] | None = None,
) -> DotsTTSAlignmentReport:
    entries = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    checkpoint: dict[str, mx.array] = {}
    duplicates = []
    for key, value in entries:
        if key in checkpoint:
            duplicates.append(key)
        else:
            checkpoint[key] = value
    parameters = tree_flatten(model.parameters(), destination={})
    model_keys = set(parameters)
    checkpoint_keys = set(checkpoint)
    shape_mismatches = []
    dtype_mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(value) for value in parameters[key].shape)
        checkpoint_shape = tuple(int(value) for value in checkpoint[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))
        tensor_dtype = expected_dtype(key) if callable(expected_dtype) else expected_dtype
        if tensor_dtype is not None and checkpoint[key].dtype != tensor_dtype:
            dtype_mismatches.append(
                (key, str(tensor_dtype), str(checkpoint[key].dtype))
            )
    return DotsTTSAlignmentReport(
        component=component,
        duplicate_checkpoint_keys=tuple(sorted(set(duplicates))),
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
        dtype_mismatches=tuple(dtype_mismatches),
    )


def _strict_load(
    component: str,
    model: _SupportsWeights,
    path: Path,
    *,
    expected_dtype: Callable[[str], mx.Dtype],
) -> DotsTTSAlignmentReport:
    weights = mx.load(str(path))
    report = align_state_dict(
        component, model, weights, expected_dtype=expected_dtype
    )
    report.require_exact()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    runtime = tree_flatten(model.parameters(), destination={})
    runtime_report = align_state_dict(
        component, model, runtime, expected_dtype=expected_dtype
    )
    report = replace(
        report,
        runtime_dtype_mismatches=runtime_report.dtype_mismatches,
    )
    report.require_exact()
    return report


def load_dots_tts_components(model_dir: str | Path) -> LoadedDotsTTSComponents:
    """Instantiate and strict-bind a base or selective-int8 artifact."""

    from .audio_vae import AudioVAE
    from .latent import LatentIO, LatentStatistics
    from .speaker import CAMPPlus, CAMPPlusConfig

    layout = validate_artifact_dir(model_dir)

    def expected(component: str) -> Callable[[str], mx.Dtype]:
        return lambda path: artifact_tensor_dtype(
            layout.artifact_config,
            component,
            path,
        )

    core = DotsTTSCoreComponents(layout.config, layout.qwen_config)
    if layout.artifact_config.quantization is not None:
        quantize_dots_tts_core(core, layout.artifact_config.quantization)
    audio_vae = AudioVAE(layout.config.vocoder)
    speaker_encoder = CAMPPlus(
        CAMPPlusConfig(embedding_size=layout.config.campplus_embedding_size)
    )
    reports = (
        _strict_load(
            "core",
            core,
            layout.model_dir / "core.safetensors",
            expected_dtype=expected("core"),
        ),
        _strict_load(
            "vocoder",
            audio_vae,
            layout.model_dir / "vocoder.safetensors",
            expected_dtype=expected("vocoder"),
        ),
        _strict_load(
            "speaker",
            speaker_encoder,
            layout.model_dir / "speaker.safetensors",
            expected_dtype=expected("speaker"),
        ),
    )
    with safe_open(
        layout.model_dir / "latent_stats.safetensors", framework="numpy"
    ) as handle:
        mean = mx.array(handle.get_tensor("mean"), dtype=mx.float32)
        variance = mx.array(handle.get_tensor("var"), dtype=mx.float32)
    latent_io = LatentIO(LatentStatistics(mean=mean, variance=variance))
    return LoadedDotsTTSComponents(
        layout=layout,
        core=core,
        audio_vae=audio_vae,
        speaker_encoder=speaker_encoder,
        latent_io=latent_io,
        reports=reports,
    )


def _validate_safetensors(
    path: Path,
    *,
    expected_keys: set[str] | None = None,
    component: str | None = None,
    artifact_config: DotsTTSArtifactConfig | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    try:
        with safe_open(path, framework="numpy") as handle:
            keys = set(handle.keys())
            if expected_keys is None and not keys:
                raise ValueError(f"safetensors file has no tensors: {path}")
            if expected_keys is not None and keys != expected_keys:
                raise ValueError(
                    f"{path.name} keys mismatch: expected={sorted(expected_keys)}, "
                    f"actual={sorted(keys)}"
                )
            if component == "core" and artifact_config is not None:
                quantization = artifact_config.quantization
                if quantization is not None:
                    expected_modules = set(quantization.quantized_paths)
                    marker_modules = {
                        key.rsplit(".", 1)[0]
                        for key in keys
                        if key.startswith("qwen.model.")
                        and key.rsplit(".", 1)[-1] in {"scales", "biases"}
                    }
                    if marker_modules != expected_modules:
                        raise ValueError(
                            "core.safetensors quantized paths differ from metadata: "
                            f"missing={sorted(expected_modules - marker_modules)}, "
                            f"unexpected={sorted(marker_modules - expected_modules)}"
                        )
                    for module_path in sorted(expected_modules):
                        required = {
                            f"{module_path}.weight",
                            f"{module_path}.scales",
                            f"{module_path}.biases",
                        }
                        missing_quantized = required - keys
                        if missing_quantized:
                            raise ValueError(
                                "core.safetensors is missing quantized tensors: "
                                f"{sorted(missing_quantized)}"
                            )
            if component is not None and artifact_config is not None:
                for key in sorted(keys):
                    expected_name = artifact_tensor_dtype_name(
                        artifact_config,
                        component,
                        key,
                    )
                    actual_name = str(handle.get_slice(key).get_dtype())
                    if actual_name != _SAFETENSORS_DTYPES[expected_name]:
                        raise ValueError(
                            f"{path.name} tensor {key} must be {expected_name}, "
                            f"got {actual_name}"
                        )
            if expected_keys == {"mean", "var"}:
                mean = handle.get_slice("mean")
                variance = handle.get_slice("var")
                if mean.get_shape() != variance.get_shape():
                    raise ValueError("latent mean and variance shapes differ")
                actual_shape = tuple(int(value) for value in mean.get_shape())
                if expected_shape is not None and actual_shape != expected_shape:
                    raise ValueError(
                        "latent mean and variance must have shape "
                        f"{expected_shape}, got {actual_shape}"
                    )
                if str(mean.get_dtype()) != "F32" or str(variance.get_dtype()) != "F32":
                    raise ValueError("latent mean and variance must be float32")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid safetensors file: {path}") from error


def validate_artifact_dir(model_dir: str | Path) -> DotsTTSArtifactLayout:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dots.tts artifact directory not found: {root}")
    top_files = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(ARTIFACT_FILES - top_files)
    unexpected = sorted(top_files - ARTIFACT_FILES)
    if missing or unexpected:
        raise ValueError(
            f"dots.tts artifact files mismatch: missing={missing}, unexpected={unexpected}"
        )
    top_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if top_directories != {"tokenizer"}:
        raise ValueError(
            "dots.tts artifact directories mismatch: "
            f"missing={sorted({'tokenizer'} - top_directories)}, "
            f"unexpected={sorted(top_directories - {'tokenizer'})}"
        )
    tokenizer_dir = root / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise ValueError("dots.tts artifact is missing tokenizer/")
    tokenizer_files = {path.name for path in tokenizer_dir.iterdir() if path.is_file()}
    if tokenizer_files != TOKENIZER_FILES:
        raise ValueError(
            "dots.tts tokenizer files mismatch: "
            f"missing={sorted(TOKENIZER_FILES - tokenizer_files)}, "
            f"unexpected={sorted(tokenizer_files - TOKENIZER_FILES)}"
        )

    config = DotsTTSConfig.from_path(root)
    qwen_config = DotsTTSQwenConfig.from_path(root)
    artifact_config = DotsTTSArtifactConfig.from_path(root)
    if root.name == "mlx-bf16":
        raise ValueError("obsolete mlx-bf16 artifacts are not accepted as mlx-base")
    expected_directory = f"mlx-{artifact_config.artifact_class}"
    if root.name.startswith("mlx-") and root.name != expected_directory:
        raise ValueError(
            f"dots.tts artifact class {artifact_config.artifact_class} requires "
            f"directory {expected_directory}, got {root.name}"
        )
    if config.mode != artifact_config.mode:
        raise ValueError("dots.tts config mode and artifact mode differ")
    if qwen_config.vocab_size <= max(
        151_668, 151_669, 151_670, 151_666, 151_671
    ):
        raise ValueError("Qwen vocabulary does not contain dots.tts special tokens")

    weight_files = tuple(
        root / name
        for name in (
            "core.safetensors",
            "vocoder.safetensors",
            "speaker.safetensors",
            "latent_stats.safetensors",
        )
    )
    for component, path in zip(
        ("core", "vocoder", "speaker"), weight_files[:-1], strict=True
    ):
        _validate_safetensors(
            path,
            component=component,
            artifact_config=artifact_config,
        )
    _validate_safetensors(
        weight_files[-1],
        expected_keys={"mean", "var"},
        component="latent_stats",
        artifact_config=artifact_config,
        expected_shape=(config.latent_dim,),
    )
    return DotsTTSArtifactLayout(
        model_dir=root,
        config=config,
        qwen_config=qwen_config,
        artifact_config=artifact_config,
        weight_files=weight_files,
        tokenizer_dir=tokenizer_dir,
    )


__all__ = [
    "ARTIFACT_FILES",
    "BASE_DTYPE_POLICY",
    "artifact_tensor_dtype",
    "artifact_tensor_dtype_name",
    "DotsTTSArtifactConfig",
    "DotsTTSArtifactLayout",
    "DotsTTSAlignmentReport",
    "DotsTTSCoreComponents",
    "DotsTTSQuantizationConfig",
    "INT8_DTYPE_POLICY",
    "SOURCE_REVISIONS",
    "TOKENIZER_FILES",
    "LoadedDotsTTSComponents",
    "align_state_dict",
    "eligible_qwen_quantization_paths",
    "load_dots_tts_components",
    "quantize_dots_tts_core",
    "storage_dtype",
    "storage_dtype_name",
    "validate_artifact_dir",
]
