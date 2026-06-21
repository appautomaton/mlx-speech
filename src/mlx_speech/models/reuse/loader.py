"""Torch -> MLX weight remap and loader for the RE-USE / SEMamba port.

Reads the upstream ``model.safetensors`` (nvidia/RE-USE) through the safetensors
numpy backend (no torch) and maps every key onto the MLX `SEMamba` parameter
tree. Two kinds of difference are handled:

1. Key renames. The encoder / decoders use torch ``nn.Sequential`` indices
   (``...dense_conv_1.0`` = Conv2d, ``.1`` = InstanceNorm2d, ``.2`` = PReLU),
   while the MLX modules name those submodules ``conv`` / ``norm`` / ``act``.
   The ``up_convN`` stages map ``.0`` -> ``conv`` similarly.

2. Conv weight layout. MLX convs are channel-last: Conv2d weight is
   ``[out, kh, kw, in]`` (torch ``[out, in, kh, kw]``) and Conv1d weight is
   ``[out, kw, in/groups]`` (torch ``[out, in/groups, kw]``). Linear, norm,
   PReLU, A_log, and D weights need no layout change.

The mamba block keys already match the MLX module names one-to-one (only the
``conv1d.weight`` layout differs), so they pass through with just that fix.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from .semamba import SEMamba

# torch Sequential index -> MLX submodule name for the ConvNormAct stages
# (dense_conv_1, dense_conv_2, and every DenseBlock entry).
_SEQ_SUFFIX = {"0": "conv", "1": "norm", "2": "act"}

# Stages whose torch key uses a plain Sequential index that we rename.
_CONVNORMACT_RE = re.compile(
    r"^(dense_encoder\.dense_conv_[12]"
    r"|(?:dense_encoder|mask_decoder|phase_decoder)\.dense_block\.dense_block\.\d+)"
    r"\.([012])\.(weight|bias)$"
)

# up_conv stages: ``up_convN.0.conv.*`` (SPConvTranspose2d), ``.1.*`` (norm),
# ``.2.weight`` (PReLU).
_UPCONV_RE = re.compile(
    r"^((?:mask_decoder|phase_decoder)\.up_conv[12])\.([012])\.(.+)$"
)


def assert_keys_match(expected_keys: set[str], actual_keys: set[str]) -> None:
    """Raise if ``actual_keys`` does not match ``expected_keys`` exactly.

    Single source of truth for the strict key-set check so the error message and
    sample size cannot drift between the loader, the converter, and the tests.
    """
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            f"key mismatch: {len(missing)} missing, {len(extra)} extra "
            f"(missing sample: {sorted(missing)[:5]}, "
            f"extra sample: {sorted(extra)[:5]})"
        )


def torch_key_to_mlx(key: str) -> str:
    """Rename a torch checkpoint key to its MLX `SEMamba` parameter path."""
    m = _CONVNORMACT_RE.match(key)
    if m:
        prefix, idx, leaf = m.groups()
        return f"{prefix}.{_SEQ_SUFFIX[idx]}.{leaf}"

    m = _UPCONV_RE.match(key)
    if m:
        prefix, idx, rest = m.groups()
        return f"{prefix}.{_SEQ_SUFFIX[idx]}.{rest}"

    # mamba blocks, final_conv, phase_conv_r/i pass through unchanged.
    return key


def _remap_weight(mlx_key: str, value: np.ndarray) -> mx.array:
    """Apply the layout transform for a single mapped weight."""
    arr = np.asarray(value)
    # Conv1d depthwise: torch [out, in/groups, kw] -> MLX [out, kw, in/groups].
    if mlx_key.endswith("conv1d.weight"):
        arr = np.transpose(arr, (0, 2, 1))
    # Conv2d: torch [out, in, kh, kw] -> MLX [out, kh, kw, in].
    elif arr.ndim == 4 and mlx_key.endswith(".weight"):
        arr = np.transpose(arr, (0, 2, 3, 1))
    return mx.array(arr)


def build_mlx_state(checkpoint: Path) -> dict[str, mx.array]:
    """Read the torch checkpoint and return the remapped MLX parameter dict."""
    weights = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not weights.exists():
        raise FileNotFoundError(f"no model.safetensors at {weights}")

    state: dict[str, mx.array] = {}
    with safe_open(str(weights), "numpy") as f:
        for key in f.keys():
            mlx_key = torch_key_to_mlx(key)
            state[mlx_key] = _remap_weight(mlx_key, f.get_tensor(key))
    return state


def load_semamba(checkpoint: Path, *, strict: bool = True) -> SEMamba:
    """Build a `SEMamba` and load the converted weights from a torch checkpoint.

    Args:
        checkpoint: directory holding ``model.safetensors`` (or the file itself).
        strict: assert the converted key set matches the model exactly.

    Returns:
        A `SEMamba` with weights loaded.
    """
    model = SEMamba()
    state = build_mlx_state(Path(checkpoint))

    if strict:
        model_keys = {k for k, _ in tree_flatten(model.parameters())}
        assert_keys_match(model_keys, set(state))

    model.update(tree_unflatten(list(state.items())))
    mx.eval(model.parameters())
    return model


def load_mlx_semamba(checkpoint: Path) -> SEMamba:
    """Load a `SEMamba` from already-converted MLX ``model.safetensors``."""
    weights = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    model = SEMamba()
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


__all__ = [
    "assert_keys_match",
    "torch_key_to_mlx",
    "build_mlx_state",
    "load_semamba",
    "load_mlx_semamba",
]
