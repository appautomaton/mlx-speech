"""Forced-alignment model family dispatch from config.json."""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_align_family(model_dir: Path) -> str:
    """Determine the alignment model family from config.json on disk."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {model_dir}. Cannot auto-detect model type."
        )

    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    thinker = payload.get("thinker_config")
    thinker_type = thinker.get("model_type", "") if isinstance(thinker, dict) else ""
    if payload.get("model_type") == "qwen3_asr" and "forced_aligner" in str(thinker_type):
        return "qwen3_forced_aligner"

    raise ValueError(
        f"{model_dir} is not a forced-alignment checkpoint "
        f"(model_type {payload.get('model_type')!r}). Supported: Qwen3-ForcedAligner."
    )
