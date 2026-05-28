"""Checkpoint loader for `AudioVAE`.

Pulls the `audio_vae.*` keys from a state dict (typically loaded from the
audio-components shard) and feeds them to `model.load_weights(...)`.

Special handling:
- per-channel-statistics: the upstream ``mean-of-means`` /
  ``std-of-means`` names use hyphens which aren't valid Python identifiers,
  so we remap them to the underscore form our MLX module exposes.
- Conv2d weights: PyTorch stores ``(out, in, kH, kW)``; MLX expects
  channel-last ``(out, kH, kW, in)``. We permute every saved Conv2d weight
  (4-D, suffix ``.weight`` under a ``.conv.`` parent).
"""

from __future__ import annotations

import mlx.core as mx

from .model import AudioVAE

_PREFIX = "audio_vae."
_PCS_PREFIX = "per_channel_statistics."

_PCS_RENAME = {
    "mean-of-means": "mean_of_means",
    "std-of-means": "std_of_means",
}


def _filter_and_rename(state: dict[str, mx.array]) -> dict[str, mx.array]:
    """Filter to `audio_vae.*` keys; rename PCS sub-keys and permute Conv2d
    weights from PyTorch (out, in, kH, kW) to MLX (out, kH, kW, in)."""
    out: dict[str, mx.array] = {}
    for k, v in state.items():
        if not k.startswith(_PREFIX):
            continue
        sub = k[len(_PREFIX):]
        if sub.startswith(_PCS_PREFIX):
            tail = sub[len(_PCS_PREFIX):]
            tail = _PCS_RENAME.get(tail, tail)
            sub = _PCS_PREFIX + tail
        # Permute 4-D Conv2d weights (saved as out,in,kH,kW → MLX out,kH,kW,in)
        if v.ndim == 4 and sub.endswith(".weight"):
            v = v.transpose(0, 2, 3, 1)
        out[sub] = v
    return out


def load_audio_vae_weights(model: AudioVAE, state: dict[str, mx.array]) -> int:
    """Load `audio_vae.*` keys into the model. Returns the number loaded."""
    sub = _filter_and_rename(state)
    if not sub:
        raise KeyError(f"No keys with prefix {_PREFIX!r} found in state dict")
    model.load_weights(list(sub.items()), strict=True)
    return len(sub)


__all__ = ["load_audio_vae_weights"]
