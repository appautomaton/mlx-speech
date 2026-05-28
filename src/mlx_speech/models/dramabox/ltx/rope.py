"""LTX-2 split RoPE for the embeddings connector (and later, the audio DiT).

The connector uses ``LTXRopeType.split`` with a NumPy-fp64 frequency grid.
The formula:

1. Build inverse frequencies via `np.power(theta, np.linspace(0, 1, dim/(2*K),
   dtype=float64))`. ``K`` is the number of positional dimensions (here ``1``
   for the 1D text sequence). Multiply by ``pi/2`` to land in the LTX
   normalized angle range. Cast to float32 at the boundary.
2. Build fractional positions `idx / max_pos[k]` for each dim ``k``, scale to
   ``[-1, 1]`` via ``x * 2 - 1``, multiply by the inverse-freq vector.
3. ``split_freqs_cis`` reshapes to ``[B, H, T, dim_head/2]``.
4. ``apply_split_rotary_emb`` rotates each pair (channel ``c`` in the first
   half of a head, channel ``c + dim_head/2`` in the second half) by the
   angle for that position-channel.

For the 1D text connector case:
- ``positional_embedding_max_pos = [4096]``
- ``positional_embedding_theta = 10000.0``
- ``inner_dim = num_heads * dim_head = 32 * 64 = 2048``
- ``seq_len = 1024`` (the padded text length)
- output cos/sin shape ``[B=1, H=32, T=1024, D/2=32]``
"""

from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx
import numpy as np


class LTXRopeType(str, Enum):
    """LTX rope variant. The DramaBox audio path uses SPLIT throughout."""

    SPLIT = "split"
    INTERLEAVED = "interleaved"


# --------------------------------------------------------------------------- #
# Frequency grid (NumPy fp64 → float32 boundary)
# --------------------------------------------------------------------------- #

def _generate_freq_grid_np(theta: float, n_pos_dims: int, inner_dim: int) -> mx.array:
    """Generate per-channel inverse-frequencies in NumPy fp64, return mx float32.

    Matches `generate_freq_grid_np` from the upstream `rope.py`. The
    ``n_pos_dims`` parameter is the number of positional dimensions
    (called ``positional_embedding_max_pos_count`` upstream); for a 1D
    sequence it is 1. ``inner_dim`` is the full inner dimension
    (``num_heads * dim_head``).
    """
    n_elem = 2 * n_pos_dims
    pow_indices = np.power(
        theta,
        np.linspace(
            math.log(1) / math.log(theta),
            math.log(theta) / math.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    out = pow_indices * math.pi / 2.0
    return mx.array(out.astype(np.float32))


# --------------------------------------------------------------------------- #
# Per-block frequency precomputation
# --------------------------------------------------------------------------- #

def precompute_split_freqs_1d(
    seq_len: int,
    inner_dim: int,
    num_heads: int,
    theta: float,
    max_pos: int,
    out_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """Pre-compute `(cos, sin)` tables for 1D split RoPE.

    Returns `(cos, sin)` both shape `[1, num_heads, seq_len, dim_head/2]` in
    `out_dtype`. The caller broadcasts on the batch axis.
    """
    n_pos_dims = 1
    inv_indices = _generate_freq_grid_np(theta, n_pos_dims, inner_dim)  # [inner_dim / (2*K)]

    # Fractional position scaled to [-1, 1]
    positions = mx.arange(seq_len, dtype=mx.float32)  # [T]
    fractional = positions / float(max_pos)  # [T]
    scaled = (fractional * 2.0) - 1.0  # [T]

    # freqs[t, c] = inv_indices[c] * scaled[t]
    freqs = mx.outer(scaled, inv_indices)  # [T, inner_dim/(2*K)]
    freqs = freqs[None, :, :]  # [1, T, inner_dim/2]

    expected = inner_dim // 2
    cur = freqs.shape[-1]
    pad_size = expected - cur
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    if pad_size > 0:
        # The padded RoPE positions become identity (cos=1, sin=0): channels
        # that don't get rotated still pass through.
        cos_pad = mx.ones((1, seq_len, pad_size), dtype=cos.dtype)
        sin_pad = mx.zeros((1, seq_len, pad_size), dtype=sin.dtype)
        cos = mx.concatenate([cos_pad, cos], axis=-1)
        sin = mx.concatenate([sin_pad, sin], axis=-1)

    # Reshape to (B, H, T, dim_head/2)
    cos = cos.reshape(1, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
    sin = sin.reshape(1, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
    return cos.astype(out_dtype), sin.astype(out_dtype)


# --------------------------------------------------------------------------- #
# RoPE from explicit positions (DiT audio path)
# --------------------------------------------------------------------------- #

def precompute_split_freqs_from_positions(
    positions: mx.array,
    inner_dim: int,
    num_heads: int,
    theta: float,
    max_pos: float,
    out_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """Pre-compute `(cos, sin)` for split RoPE from explicit positional bounds.

    Args:
        positions: ``[B, 1, T, 2]`` (start, end) pairs in seconds — produced
            by `AudioPatchifier.get_patch_grid_bounds(shape)`. The "1" axis
            is the number of positional dimensions (= 1 for audio-only).
        inner_dim: full inner dim (num_heads * dim_head).
        num_heads: H.
        theta: positional embedding base (10_000 for audio).
        max_pos: scalar max-position used to normalize the timing into the
            ``[-1, 1]`` RoPE range. For DramaBox audio this is ``20`` seconds
            (per `audio_positional_embedding_max_pos = [20]`).
        out_dtype: target dtype for the returned tables.

    Returns:
        ``(cos, sin)`` both of shape ``(B, num_heads, T, dim_head/2)``.

    Mirrors `precompute_freqs_cis(use_middle_indices_grid=True)` from
    `.references/DramaBox/ltx2/ltx_core/model/transformer/rope.py`:

        middle = (start + end) / 2                  # (B, 1, T)
        fractional = middle[:, 0] / max_pos         # (B, T)
        scaled = fractional * 2 - 1                 # (B, T)
        freqs = scaled[:, :, None] * inv_indices    # (B, T, inner_dim/2)
        cos = cos(freqs);  sin = sin(freqs)
        cos = cos.reshape(B, T, H, D/2).transpose → (B, H, T, D/2)
    """
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError(
            f"positions must have shape [B, n_pos_dims, T, 2]; got {positions.shape}"
        )
    n_pos_dims = positions.shape[1]
    if n_pos_dims != 1:
        raise NotImplementedError(
            f"precompute_split_freqs_from_positions only supports 1D (audio-only) positions; "
            f"got n_pos_dims={n_pos_dims}"
        )
    inv_indices = _generate_freq_grid_np(theta, n_pos_dims, inner_dim)  # [inner_dim/2]

    # use_middle_indices_grid=True: average start + end
    start = positions[..., 0]  # (B, 1, T)
    end = positions[..., 1]
    middle = (start + end) / 2.0  # (B, 1, T)
    middle = middle[:, 0]  # (B, T) — drop the n_pos_dims axis

    # Normalize to [-1, 1] using max_pos
    fractional = middle / float(max_pos)  # (B, T)
    scaled = fractional * 2.0 - 1.0  # (B, T)

    # freqs[b, t, c] = inv_indices[c] * scaled[b, t]
    freqs = scaled[:, :, None] * inv_indices[None, None, :]  # (B, T, inner_dim/2)

    expected = inner_dim // 2
    cur = freqs.shape[-1]
    pad_size = expected - cur
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    if pad_size > 0:
        B, T_, _ = freqs.shape
        cos_pad = mx.ones((B, T_, pad_size), dtype=cos.dtype)
        sin_pad = mx.zeros((B, T_, pad_size), dtype=sin.dtype)
        cos = mx.concatenate([cos_pad, cos], axis=-1)
        sin = mx.concatenate([sin_pad, sin], axis=-1)

    # (B, T, inner_dim/2) → (B, T, H, dim_head/2) → (B, H, T, dim_head/2)
    B, T_, _ = cos.shape
    cos = cos.reshape(B, T_, num_heads, -1).transpose(0, 2, 1, 3)
    sin = sin.reshape(B, T_, num_heads, -1).transpose(0, 2, 1, 3)
    return cos.astype(out_dtype), sin.astype(out_dtype)


# --------------------------------------------------------------------------- #
# Apply split RoPE
# --------------------------------------------------------------------------- #

def apply_split_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply split RoPE to ``x``.

    Args:
        x: ``(B, T, inner_dim)`` — the projected q or k tensor BEFORE reshape
            to heads. We reshape into ``(B, H, T, D)``, rotate, and return
            back to ``(B, T, inner_dim)``.
        cos: ``(B, H, T, D/2)``
        sin: ``(B, H, T, D/2)``

    Each head's last dim ``D`` is split into ``(2, D/2)`` (first/second
    half). The pair `(c, c + D/2)` is rotated by `angle = freqs[t, c]`:

        new_first[c]  = first[c]  * cos[c] - second[c] * sin[c]
        new_second[c] = second[c] * cos[c] + first[c]  * sin[c]
    """
    if x.ndim == 3:
        B, T, _ = x.shape
        H = cos.shape[1]
        D = (cos.shape[-1]) * 2
        # reshape (B, T, H*D) → (B, T, H, D) → (B, H, T, D)
        x = x.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"apply_split_rope expects (B, T, inner_dim); got ndim={x.ndim}")

    # Split last dim D into two halves of size D/2
    half = D // 2
    first = x[..., :half]
    second = x[..., half:]

    new_first = first * cos - second * sin
    new_second = second * cos + first * sin

    out = mx.concatenate([new_first, new_second], axis=-1)  # (B, H, T, D)
    # Back to (B, T, H*D)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    return out


__all__ = [
    "LTXRopeType",
    "precompute_split_freqs_1d",
    "precompute_split_freqs_from_positions",
    "apply_split_rope",
]
