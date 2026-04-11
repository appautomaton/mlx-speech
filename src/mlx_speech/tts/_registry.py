"""TTS model family dispatch from config.json."""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_tts_family(model_dir: Path) -> str:
    """Determine the TTS model family from config.json on disk."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {model_dir}. "
            "Cannot auto-detect model type."
        )

    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    model_type = payload.get("model_type", "")

    if model_type == "fish_qwen3_omni":
        return "fish_s2_pro"
    if model_type == "vibevoice":
        return "vibevoice"
    if model_type == "audiodit":
        return "longcat"
    if model_type == "step1":
        return "step_audio"
    if model_type == "moss_tts_delay":
        dir_lower = str(model_dir).lower()
        if "sound_effect" in dir_lower or "sound-effect" in dir_lower:
            return "moss_sound_effect"
        n_vq = int(payload.get("n_vq", 0))
        return "moss_local" if n_vq > 20 else "moss_delay"

    raise ValueError(
        f"Unknown TTS model_type {model_type!r} in {model_dir}. "
        "Supported: fish_qwen3_omni, vibevoice, audiodit, step1, moss_tts_delay."
    )
