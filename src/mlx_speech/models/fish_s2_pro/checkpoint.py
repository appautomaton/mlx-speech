from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

import mlx.core as mx
from safetensors import safe_open

from .config import FishS2ProConfig


@dataclass
class FishS2ProCheckpoint:
    """Loaded Fish S2 Pro checkpoint."""

    state_dict: Dict[str, mx.array]
    config: FishS2ProConfig


def load_fish_s2_pro_checkpoint(
    model_dir: str,
    strict: bool = True,
) -> FishS2ProCheckpoint:
    """Load Fish S2 Pro checkpoint from directory.

    Args:
        model_dir: Path to checkpoint directory
        strict: Whether to require all keys

    Returns:
        FishS2ProCheckpoint with state_dict and config
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

    state_dict: Dict[str, mx.array] = {}

    for safetensor_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(safetensor_file, framework="mlx") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config = FishS2ProConfig(
            vocab_size=config_data.get("vocab_size", 4096),
            num_layers=config_data.get("num_layers", 30),
            max_position_embeddings=config_data.get("max_position_embeddings", 8192),
            num_heads=config_data.get("num_heads", 16),
            slow_ar_dim=config_data.get("slow_ar_dim", 2048),
            fast_ar_dim=config_data.get("fast_ar_dim", 1024),
            num_codebooks=config_data.get("num_codebooks", 10),
            model_dir=str(model_dir),
        )
    else:
        config = FishS2ProConfig(model_dir=str(model_dir))

    return FishS2ProCheckpoint(state_dict=state_dict, config=config)


def load_checkpoint_into_model(
    model,
    checkpoint: FishS2ProCheckpoint,
    strict: bool = True,
) -> Dict[str, any]:
    """Load checkpoint weights into model.

    Args:
        model: DualARTransformer
        checkpoint: FishS2ProCheckpoint
        strict: Whether to require all keys

    Returns:
        Report dict with loading details
    """
    model_params = dict(model.named_parameters())
    state_dict = checkpoint.state_dict

    loaded = {}
    missing = {}

    for key, param in model_params.items():
        if key in state_dict:
            param.value = state_dict[key]
            loaded[key] = state_dict[key].shape
        else:
            missing[key] = param.shape

    return {
        "loaded": loaded,
        "missing": missing,
        "is_exact_match": len(missing) == 0,
    }
