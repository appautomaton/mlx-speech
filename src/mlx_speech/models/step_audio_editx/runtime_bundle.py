"""Step-Audio-EditX self-contained runtime bundle contract."""

from __future__ import annotations

from pathlib import Path


STEP_AUDIO_EDITX_RUNTIME_FILES = frozenset(
    {
        "campplus-config.json",
        "campplus.safetensors",
        "config.json",
        "flow-conditioner-config.json",
        "flow-conditioner.safetensors",
        "flow-model-config.json",
        "flow-model.safetensors",
        "frontend-config.json",
        "hift-config.json",
        "hift.safetensors",
        "model.safetensors",
        "step-audio-tokenizer-assets.safetensors",
        "step-audio-tokenizer-config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vq02-config.json",
        "vq02.safetensors",
        "vq06-config.json",
        "vq06.safetensors",
    }
)


def validate_step_audio_editx_runtime_bundle(model_dir: str | Path) -> Path:
    resolved = Path(model_dir)
    missing = tuple(
        sorted(
            name
            for name in STEP_AUDIO_EDITX_RUNTIME_FILES
            if not (resolved / name).is_file()
        )
    )
    if missing:
        raise FileNotFoundError(
            f"Incomplete Step-Audio runtime bundle in {resolved}: missing {missing}."
        )
    return resolved


__all__ = [
    "STEP_AUDIO_EDITX_RUNTIME_FILES",
    "validate_step_audio_editx_runtime_bundle",
]
