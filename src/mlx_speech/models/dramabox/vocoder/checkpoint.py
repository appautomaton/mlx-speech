"""Checkpoint loader for `VocoderWithBWE`.

Filters keys under ``vocoder.*`` prefix, strips it, and applies per-op
remaps:

- Conv1d weights: PyTorch ``(out, in, K)`` → MLX ``(out, K, in)``
- ConvTranspose1d weights: PyTorch ``(in, out, K)`` → MLX ``(out, K, in)``
- ``mel_stft.stft_fn.{forward,inverse}_basis``: kept as-is (the inner code
  permutes them at forward time to match MLX conv weight layout).
"""

from __future__ import annotations

import mlx.core as mx

from .vocoder_with_bwe import VocoderWithBWE

_PREFIX = "vocoder."

# ConvTranspose1d keys (PyTorch (in, out, K) → MLX (out, K, in)) live under
# the `ups.N.weight` path inside both `vocoder` and `bwe_generator`.
_CONVTRANSPOSE_SUFFIXES = ("ups.0.weight", "ups.1.weight", "ups.2.weight",
                            "ups.3.weight", "ups.4.weight", "ups.5.weight")


def _is_convtranspose_key(sub: str) -> bool:
    return any(sub.endswith(s) for s in _CONVTRANSPOSE_SUFFIXES) and ".ups." in sub


def _is_stft_basis_key(sub: str) -> bool:
    return sub.endswith("forward_basis") or sub.endswith("inverse_basis")


def _filter_and_permute(state: dict[str, mx.array]) -> dict[str, mx.array]:
    """Filter to `vocoder.*` keys and permute Conv weights.

    Conv1d  weights (3-D, .weight): (out, in, K) → (out, K, in)   permute (0, 2, 1)
    ConvT1d weights (3-D, .weight): (in, out, K) → (out, K, in)   permute (1, 2, 0)
    STFT basis (3-D, named forward_basis/inverse_basis): keep as-is — the
        STFT helper does its own permute at forward time.
    """
    out: dict[str, mx.array] = {}
    for k, v in state.items():
        if not k.startswith(_PREFIX):
            continue
        sub = k[len(_PREFIX):]
        if v.ndim == 3 and sub.endswith(".weight"):
            if _is_convtranspose_key(sub):
                v = v.transpose(1, 2, 0)
            else:
                v = v.transpose(0, 2, 1)
        out[sub] = v
    return out


def load_vocoder_with_bwe_weights(model: VocoderWithBWE, state: dict[str, mx.array]) -> int:
    """Load ``vocoder.*`` keys into `model`. Returns the number of loaded keys."""
    sub = _filter_and_permute(state)
    if not sub:
        raise KeyError(f"No keys with prefix {_PREFIX!r} found in state dict")
    model.load_weights(list(sub.items()), strict=True)
    return len(sub)


__all__ = ["load_vocoder_with_bwe_weights"]
