"""Selective-scan (SSM) recurrence in pure MLX.

Mirrors `.references/mamba_ssm/selective_scan_interface.py:selective_scan_ref`
exactly for the real-valued, input-dependent (variable) B/C case that SEMamba
uses. This is a faithful port of the upstream recurrence, not a re-derivation.

Layout (matches the upstream reference):
- `u`, `delta`, `z`: `[B, d_inner, L]`
- `A`: `[d_inner, d_state]` (already negative; the caller computes
  `A = -exp(A_log)` and passes it as-is)
- variable `B`, `C`: `[B, d_state, L]`
- `D`: `[d_inner]`
- `delta_bias`: `[d_inner]`

Recurrence over the L axis:
    delta   = softplus(dt + delta_bias)          # when delta_softplus
    deltaA  = exp(delta[..., None] * A)           # [B, d_inner, L, d_state]
    deltaBu = delta[..., None] * B * u[..., None] # variable-B
    x_t     = deltaA_t * x_{t-1} + deltaBu_t      # state [B, d_inner, d_state]
    y_t     = einsum('bdn,bn->bd', x_t, C_t)
    out     = y + u * D                           # when D is given
    out     = out * silu(z)                       # when z is given
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def selective_scan(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array | None = None,
    z: mx.array | None = None,
    delta_bias: mx.array | None = None,
    delta_softplus: bool = True,
) -> mx.array:
    """Run the SSM selective scan with variable (input-dependent) B and C.

    Args:
        u: input `[B, d_inner, L]`.
        delta: pre-activation dt `[B, d_inner, L]`.
        A: state matrix `[d_inner, d_state]`, already negative.
        B: variable input matrix `[B, d_state, L]`.
        C: variable output matrix `[B, d_state, L]`.
        D: optional skip term `[d_inner]`.
        z: optional gate `[B, d_inner, L]`, applied as `silu(z)`.
        delta_bias: optional bias `[d_inner]` added to `delta` before softplus.
        delta_softplus: apply softplus to `delta` (upstream default True).

    Returns:
        y: `[B, d_inner, L]`.
    """
    # Force contiguous float32. Callers may pass reversed-stride views
    # (`t[:, :, ::-1]`); some MLX 0.31 elementwise ops (e.g. silu) give wrong
    # results on such views, so materialize before any math.
    def f32(t: mx.array) -> mx.array:
        return mx.contiguous(t.astype(mx.float32))

    u = f32(u)
    delta = f32(delta)
    A = f32(A)
    B = f32(B)
    C = f32(C)

    if delta_bias is not None:
        delta = delta + delta_bias.astype(mx.float32)[None, :, None]
    if delta_softplus:
        delta = nn.softplus(delta)

    batch, dim, length = u.shape
    dstate = A.shape[1]

    # deltaA: exp(delta[b,d,l] * A[d,n]) -> [B, d_inner, L, d_state]
    deltaA = mx.exp(delta[..., None] * A[None, :, None, :])
    # deltaB_u: delta[b,d,l] * B[b,n,l] * u[b,d,l] -> [B, d_inner, L, d_state]
    # B is [B, d_state, L]; bring it to [B, 1, L, d_state] to broadcast over dim.
    B_t = mx.transpose(B, (0, 2, 1))[:, None, :, :]  # [B, 1, L, d_state]
    deltaB_u = delta[..., None] * B_t * u[..., None]  # [B, d_inner, L, d_state]

    # C as [B, L, d_state] for per-step einsum('bdn,bn->bd').
    C_t = mx.transpose(C, (0, 2, 1))  # [B, L, d_state]

    x = mx.zeros((batch, dim, dstate), dtype=mx.float32)
    ys = []
    for i in range(length):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]  # [B, d_inner, d_state]
        # y[b,d] = sum_n x[b,d,n] * C[b,i,n]
        y = mx.sum(x * C_t[:, None, i, :], axis=-1)  # [B, d_inner]
        ys.append(y)
    y = mx.stack(ys, axis=2)  # [B, d_inner, L]

    if D is not None:
        y = y + u * D.astype(mx.float32)[None, :, None]
    if z is not None:
        y = y * nn.silu(f32(z))
    return y
