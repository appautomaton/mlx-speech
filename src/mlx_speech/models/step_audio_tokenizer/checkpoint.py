"""Asset loading helpers for the Step-Audio tokenizer family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...checkpoints import (
    LoadedTorchArchiveStateDict,
    get_stepfun_v4_layouts,
    load_torch_archive_state_dict,
)
from .config import DEFAULT_FUNASR_MODEL_ID, StepAudioTokenizerConfig


@dataclass(frozen=True)
class StepAudioTokenizerAssets:
    """Loaded local Step-Audio tokenizer asset bundle."""

    model_dir: Path
    config: StepAudioTokenizerConfig
    linguistic_tokenizer_path: Path
    semantic_tokenizer_path: Path
    funasr_model_dir: Path
    funasr_config_path: Path
    funasr_checkpoint_path: Path
    linguistic_codebook: np.ndarray


def resolve_step_audio_tokenizer_model_dir(
    model_dir: str | Path | None = None,
) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    layout = get_stepfun_v4_layouts().step_audio_tokenizer
    if layout.original_dir.exists():
        return layout.original_dir
    raise FileNotFoundError(
        "No local Step-Audio tokenizer asset directory found at "
        f"{layout.original_dir}."
    )


def _resolve_funasr_dir(model_dir: Path, funasr_model_id: str) -> Path:
    candidate = model_dir / funasr_model_id
    if candidate.exists():
        return candidate

    namespace, _, repo_name = funasr_model_id.partition("/")
    candidate = model_dir / namespace / repo_name
    if candidate.exists():
        return candidate

    if namespace:
        namespace_dir = model_dir / namespace
        if namespace_dir.exists():
            matches = sorted(path for path in namespace_dir.iterdir() if path.is_dir())
            if len(matches) == 1:
                return matches[0]

    matches = sorted(path for path in model_dir.rglob("config.yaml") if path.is_file())
    if len(matches) == 1:
        return matches[0].parent

    raise FileNotFoundError(
        "Unable to resolve the Step-Audio FunASR asset directory under "
        f"{model_dir} for {funasr_model_id!r}."
    )


def load_step_audio_tokenizer_assets(
    model_dir: str | Path | None = None,
) -> StepAudioTokenizerAssets:
    resolved_dir = resolve_step_audio_tokenizer_model_dir(model_dir)
    linguistic_tokenizer_path = resolved_dir / "linguistic_tokenizer.npy"
    semantic_tokenizer_path = resolved_dir / "speech_tokenizer_v1.onnx"

    if not linguistic_tokenizer_path.exists():
        raise FileNotFoundError(f"Missing Step-Audio linguistic tokenizer: {linguistic_tokenizer_path}")
    if not semantic_tokenizer_path.exists():
        raise FileNotFoundError(f"Missing Step-Audio semantic tokenizer: {semantic_tokenizer_path}")

    linguistic_codebook = np.load(linguistic_tokenizer_path)
    if linguistic_codebook.ndim != 2:
        raise ValueError(
            "Expected linguistic_tokenizer.npy to be rank-2, got "
            f"{linguistic_codebook.shape}."
        )

    funasr_model_dir = _resolve_funasr_dir(resolved_dir, DEFAULT_FUNASR_MODEL_ID)
    funasr_config_path = funasr_model_dir / "config.yaml"
    funasr_checkpoint_path = funasr_model_dir / "model.pt"
    if not funasr_config_path.exists():
        raise FileNotFoundError(f"Missing FunASR config: {funasr_config_path}")
    if not funasr_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing FunASR checkpoint: {funasr_checkpoint_path}")

    config = StepAudioTokenizerConfig.from_loaded_assets(
        vq02_codebook_size=int(linguistic_codebook.shape[0]),
        extra={"funasr_model_dir": str(funasr_model_dir)},
    )
    return StepAudioTokenizerAssets(
        model_dir=resolved_dir,
        config=config,
        linguistic_tokenizer_path=linguistic_tokenizer_path,
        semantic_tokenizer_path=semantic_tokenizer_path,
        funasr_model_dir=funasr_model_dir,
        funasr_config_path=funasr_config_path,
        funasr_checkpoint_path=funasr_checkpoint_path,
        linguistic_codebook=linguistic_codebook,
    )


def load_step_audio_funasr_checkpoint(
    model_dir: str | Path | None = None,
) -> LoadedTorchArchiveStateDict:
    assets = load_step_audio_tokenizer_assets(model_dir)
    return load_torch_archive_state_dict(assets.funasr_checkpoint_path)
