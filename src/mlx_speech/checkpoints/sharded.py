"""Helpers for sharded safetensors checkpoints."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


INDEX_FILENAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ShardedCheckpointIndex:
    """Parsed contents of `model.safetensors.index.json`."""

    model_dir: Path
    index_path: Path
    metadata: dict[str, Any]
    weight_map: dict[str, str]

    @classmethod
    def from_directory(
        cls,
        model_dir: str | Path,
        index_filename: str = INDEX_FILENAME,
    ) -> "ShardedCheckpointIndex":
        resolved_dir = Path(model_dir)
        index_path = resolved_dir / index_filename
        if not index_path.exists():
            raise FileNotFoundError(f"Checkpoint index not found: {index_path}")

        with index_path.open(encoding="utf-8") as f:
            payload = json.load(f)

        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")

        return cls(
            model_dir=resolved_dir,
            index_path=index_path,
            metadata=dict(payload.get("metadata", {})),
            weight_map={str(k): str(v) for k, v in weight_map.items()},
        )

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.weight_map.values()))

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(self.model_dir / shard_name for shard_name in self.shard_names)


@dataclass(frozen=True)
class LoadedStateDict:
    """Loaded state dict plus provenance information."""

    model_dir: Path
    files: tuple[Path, ...]
    weights: dict[str, mx.array]
    index: ShardedCheckpointIndex | None = None


def _discover_non_sharded_files(model_dir: Path) -> tuple[Path, ...]:
    files = tuple(sorted(model_dir.glob("model*.safetensors")))
    if files:
        return files
    return tuple(sorted(model_dir.glob("*.safetensors")))


def load_state_dict(
    model_dir: str | Path,
    index_filename: str = INDEX_FILENAME,
) -> LoadedStateDict:
    """Load `.safetensors` weights from a directory."""

    resolved_dir = Path(model_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {resolved_dir}")

    index_path = resolved_dir / index_filename
    index: ShardedCheckpointIndex | None = None
    if index_path.exists():
        index = ShardedCheckpointIndex.from_directory(resolved_dir, index_filename=index_filename)
        files = index.shard_paths
    else:
        files = _discover_non_sharded_files(resolved_dir)

    if not files:
        raise FileNotFoundError(
            f"No `.safetensors` files found under checkpoint directory: {resolved_dir}"
        )

    weights: dict[str, mx.array] = {}
    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint shard not found: {file_path}")
        shard_weights = mx.load(str(file_path))
        for key, value in shard_weights.items():
            if key in weights:
                raise ValueError(f"Duplicate tensor key encountered while loading {resolved_dir}: {key}")
            weights[key] = value

    return LoadedStateDict(
        model_dir=resolved_dir,
        files=files,
        weights=weights,
        index=index,
    )


def summarize_prefixes(weights: dict[str, mx.array], depth: int = 2) -> list[tuple[str, int]]:
    """Count weight prefixes to make checkpoint inspection easier."""

    counter: Counter[str] = Counter()
    for key in weights:
        parts = key.split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else key
        counter[prefix] += 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))
