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

import os

import mlx.core as mx
import mlx.nn as nn

_CUSTOM_METAL_AVAILABLE = (
    os.environ.get("MLX_SPEECH_DISABLE_CUSTOM_METAL") != "1" and mx.metal.is_available()
)

# One thread per (batch, channel) carries its state in registers, so d_state
# must fit the fixed 32-slot arrays below.
_FUSED_SCAN_MAX_DSTATE = 32

_FUSED_SCAN_KERNEL = mx.fast.metal_kernel(
    name="selective_scan_fused",
    input_names=["u", "delta", "A", "B_t", "C_t", "L_arr", "D_arr", "N_arr"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        int L = L_arr[0];
        int D = D_arr[0];
        int N = N_arr[0];
        int b = tid / D;
        int d = tid % D;

        int u_base = tid * L;
        int a_base = d * N;
        int bc_base = b * L * N;

        float a_reg[32];
        float x[32];
        for (int n = 0; n < N; ++n) {
            a_reg[n] = A[a_base + n];
            x[n] = 0.0f;
        }

        for (int i = 0; i < L; ++i) {
            float dt = delta[u_base + i];
            float dt_u = dt * u[u_base + i];
            int bc_idx = bc_base + i * N;
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                float da = metal::exp(dt * a_reg[n]);
                float s = da * x[n] + dt_u * B_t[bc_idx + n];
                x[n] = s;
                acc += s * C_t[bc_idx + n];
            }
            out[u_base + i] = acc;
        }
    """,
)


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
        delta = mx.contiguous(nn.softplus(delta))

    batch, dim, length = u.shape
    dstate = A.shape[1]

    if (
        _CUSTOM_METAL_AVAILABLE
        and dstate <= _FUSED_SCAN_MAX_DSTATE
        and batch * dim > 0
        and length > 0
    ):
        B_t = mx.contiguous(mx.transpose(B, (0, 2, 1)))  # [B, L, d_state]
        C_t = mx.contiguous(mx.transpose(C, (0, 2, 1)))  # [B, L, d_state]
        shapes = (
            mx.array([length], dtype=mx.int32),
            mx.array([dim], dtype=mx.int32),
            mx.array([dstate], dtype=mx.int32),
        )
        y = _FUSED_SCAN_KERNEL(
            inputs=[u, delta, A, B_t, C_t, *shapes],
            grid=(batch * dim, 1, 1),
            threadgroup=(min(256, batch * dim), 1, 1),
            output_shapes=[(batch, dim, length)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        deltaA = mx.exp(delta[..., None] * A[None, :, None, :])
        B_t = mx.transpose(B, (0, 2, 1))[:, None, :, :]
        deltaB_u = delta[..., None] * B_t * u[..., None]
        C_t = mx.transpose(C, (0, 2, 1))
        x = mx.zeros((batch, dim, dstate), dtype=mx.float32)
        ys = []
        for i in range(length):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            ys.append(mx.sum(x * C_t[:, None, i, :], axis=-1))
        y = mx.stack(ys, axis=2)

    if D is not None:
        y = y + u * D.astype(mx.float32)[None, :, None]
    if z is not None:
        y = y * nn.silu(f32(z))
    return y
