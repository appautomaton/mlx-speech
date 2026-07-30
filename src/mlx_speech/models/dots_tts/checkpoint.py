"""Strict native artifact contract for converted dots.tts checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
BF16_DTYPE_POLICY = {
    "activations": "bfloat16",
    "qwen": "bfloat16",
    "semantic_encoder": "bfloat16",
    "dit": "bfloat16",
    "vocoder": "bfloat16",
    "speaker": "bfloat16",
}
INT8_DTYPE_POLICY = {**BF16_DTYPE_POLICY, "qwen": "int8"}


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
            module_types=tuple(str(value) for value in payload.get("module_types", ())),
            path_prefixes=tuple(str(value) for value in payload.get("path_prefixes", ())),
            quantized_paths=tuple(str(value) for value in payload.get("quantized_paths", ())),
        )
        if config.bits != 8 or config.group_size != 64 or config.mode != "affine":
            raise ValueError("dots.tts int8 requires affine 8-bit groups of 64")
        if config.module_types != ("Linear", "Embedding"):
            raise ValueError("dots.tts int8 may target only Linear and Embedding modules")
        if config.path_prefixes != ("llm.",):
            raise ValueError("dots.tts int8 may target only the Qwen llm.* trunk")
        if not config.quantized_paths:
            raise ValueError("dots.tts int8 metadata must name every quantized path")
        if any(not path.startswith("llm.") for path in config.quantized_paths):
            raise ValueError("dots.tts quantized paths must stay under llm.*")
        if len(set(config.quantized_paths)) != len(config.quantized_paths):
            raise ValueError("dots.tts quantized paths must be unique")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "module_types": list(self.module_types),
            "path_prefixes": list(self.path_prefixes),
            "quantized_paths": list(self.quantized_paths),
        }


@dataclass(frozen=True)
class DotsTTSArtifactConfig:
    schema_version: int
    model_family: str
    variant: Literal["soar", "mf"]
    mode: Literal["flow_matching", "meanflow"]
    precision: Literal["bf16", "int8"]
    source_repo_id: str
    source_resolved_repo_id: str
    source_revision: str
    source_manifest_sha256: str
    dtype_policy: dict[str, str]
    quantization: DotsTTSQuantizationConfig | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSArtifactConfig":
        required = {
            "schema_version",
            "model_family",
            "variant",
            "mode",
            "precision",
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
            precision=str(payload["precision"]),  # type: ignore[arg-type]
            source_repo_id=str(source.get("repo_id", "")),
            source_resolved_repo_id=str(source.get("resolved_repo_id", "")),
            source_revision=str(source.get("revision", "")),
            source_manifest_sha256=str(source.get("manifest_sha256", "")),
            dtype_policy={
                str(key): str(value) for key, value in payload["dtype_policy"].items()
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
        if self.precision not in {"bf16", "int8"}:
            raise ValueError(f"unsupported dots.tts precision: {self.precision}")
        expected_policy = (
            BF16_DTYPE_POLICY if self.precision == "bf16" else INT8_DTYPE_POLICY
        )
        if self.dtype_policy != expected_policy:
            raise ValueError("dots.tts dtype policy is inconsistent with precision")
        if self.precision == "bf16" and self.quantization is not None:
            raise ValueError("BF16 dots.tts artifacts cannot carry quantization metadata")
        if self.precision == "int8" and self.quantization is None:
            raise ValueError("int8 dots.tts artifacts require quantization metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "variant": self.variant,
            "mode": self.mode,
            "precision": self.precision,
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


def _validate_safetensors(path: Path, *, expected_keys: set[str] | None = None) -> None:
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
            if expected_keys == {"mean", "var"}:
                mean = handle.get_slice("mean")
                variance = handle.get_slice("var")
                if mean.get_shape() != variance.get_shape():
                    raise ValueError("latent mean and variance shapes differ")
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
    for path in weight_files[:-1]:
        _validate_safetensors(path)
    _validate_safetensors(weight_files[-1], expected_keys={"mean", "var"})
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
    "BF16_DTYPE_POLICY",
    "DotsTTSArtifactConfig",
    "DotsTTSArtifactLayout",
    "DotsTTSQuantizationConfig",
    "INT8_DTYPE_POLICY",
    "SOURCE_REVISIONS",
    "TOKENIZER_FILES",
    "validate_artifact_dir",
]
