"""ASR model family dispatch from config.json."""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_asr_family(model_dir: Path) -> str:
    """Determine the ASR model family from config.json on disk."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {model_dir}. "
            "Cannot auto-detect model type."
        )

    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    model_type = payload.get("model_type", "")

    if model_type == "cohere_asr":
        return "cohere"

    raise ValueError(
        f"Unknown ASR model_type {model_type!r} in {model_dir}. "
        "Supported: cohere_asr."
    )
