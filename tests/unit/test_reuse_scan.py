"""Unit tests for the MLX selective scan against a numpy port of upstream.

The reference `selective_scan_np` is a direct transcription of
`.references/mamba_ssm/selective_scan_interface.py:selective_scan_ref` for the
real-valued, variable (input-dependent) B/C case that SEMamba uses. No torch is
imported; numpy expresses the recurrence so the MLX port can be checked against
the exact upstream math.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from mlx_speech.models.reuse.mamba.scan import (
    selective_scan,
    selective_scan_reverse,
)

# SEMamba per-direction dims (see slice-001 notes): d_inner=256, d_state=16.
# Tests use smaller dims for speed; the math is identical.
B_DIM = 2
D_INNER = 6
D_STATE = 4
L = 7


def _softplus_np(x: np.ndarray) -> np.ndarray:
    # Stable softplus: log1p(exp(-|x|)) + max(x, 0).
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def selective_scan_np(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray | None = None,
    z: np.ndarray | None = None,
    delta_bias: np.ndarray | None = None,
    delta_softplus: bool = True,
) -> np.ndarray:
    """Numpy port of selective_scan_ref (real, variable-B, variable-C).

    u, delta, z: [B, d_inner, L]; A: [d_inner, d_state];
    B, C: [B, d_state, L]; D: [d_inner]; delta_bias: [d_inner].
    Returns y: [B, d_inner, L].
    """
    u = u.astype(np.float64)
    delta = delta.astype(np.float64)
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)
    if delta_bias is not None:
        delta = delta + delta_bias.astype(np.float64)[None, :, None]
    if delta_softplus:
        delta = _softplus_np(delta)
    batch, dim = u.shape[0], A.shape[0]
    dstate = A.shape[1]
    # deltaA: exp(einsum('bdl,dn->bdln', delta, A))
    deltaA = np.exp(np.einsum("bdl,dn->bdln", delta, A))
    # deltaB_u (variable B, B.dim()==3): einsum('bdl,bnl,bdl->bdln', delta, B, u)
    deltaB_u = np.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = np.zeros((batch, dim, dstate), dtype=np.float64)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        # variable C, C.dim()==3: einsum('bdn,bn->bd', x, C[:, :, i])
        y = np.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)
    y = np.stack(ys, axis=2)
    out = y if D is None else y + u * D.astype(np.float64)[:, None]
    if z is not None:
        out = out * _silu_np(z.astype(np.float64))
    return out


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_inputs(seed: int = 0):
    rng = _rng(seed)
    u = rng.standard_normal((B_DIM, D_INNER, L)).astype(np.float32)
    delta = rng.standard_normal((B_DIM, D_INNER, L)).astype(np.float32)
    # A is negative (caller computes A = -exp(A_log)).
    A = -np.exp(rng.standard_normal((D_INNER, D_STATE)).astype(np.float32))
    B = rng.standard_normal((B_DIM, D_STATE, L)).astype(np.float32)
    C = rng.standard_normal((B_DIM, D_STATE, L)).astype(np.float32)
    D = rng.standard_normal((D_INNER,)).astype(np.float32)
    z = rng.standard_normal((B_DIM, D_INNER, L)).astype(np.float32)
    delta_bias = rng.standard_normal((D_INNER,)).astype(np.float32)
    return u, delta, A, B, C, D, z, delta_bias


def _to_mx(*arrays):
    return [mx.array(a) for a in arrays]


def _max_abs_diff(got: mx.array, ref: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))


def test_softplus_with_nonzero_delta_bias():
    u, delta, A, B, C, D, z, delta_bias = _make_inputs(1)
    ref = selective_scan_np(
        u, delta, A, B, C, D=None, z=None,
        delta_bias=delta_bias, delta_softplus=True,
    )
    mu, md, mA, mB, mC, mbias = _to_mx(u, delta, A, B, C, delta_bias)
    got = selective_scan(
        mu, md, mA, mB, mC, D=None, z=None,
        delta_bias=mbias, delta_softplus=True,
    )
    assert got.shape == (B_DIM, D_INNER, L)
    assert _max_abs_diff(got, ref) < 1e-4


def test_negative_A_decaying_deltaA():
    # With negative A and positive softplus(delta), deltaA = exp(delta*A) < 1,
    # so the recurrence decays. Verify the port matches upstream exactly here.
    u, delta, A, B, C, _D, _z, _bias = _make_inputs(2)
    assert np.all(A < 0.0)
    ref = selective_scan_np(u, delta, A, B, C, delta_softplus=True)
    mu, md, mA, mB, mC = _to_mx(u, delta, A, B, C)
    got = selective_scan(mu, md, mA, mB, mC, delta_softplus=True)
    assert _max_abs_diff(got, ref) < 1e-4


def test_variable_B_and_C():
    # Distinct per-(state, time) B and C exercise the input-dependent path.
    u, delta, A, B, C, _D, _z, _bias = _make_inputs(3)
    assert not np.allclose(B, C)
    ref = selective_scan_np(u, delta, A, B, C, delta_softplus=True)
    mu, md, mA, mB, mC = _to_mx(u, delta, A, B, C)
    got = selective_scan(mu, md, mA, mB, mC, delta_softplus=True)
    assert _max_abs_diff(got, ref) < 1e-4


def test_d_skip_term_with_and_without():
    u, delta, A, B, C, D, _z, _bias = _make_inputs(4)
    mu, md, mA, mB, mC, mD = _to_mx(u, delta, A, B, C, D)

    ref_no_d = selective_scan_np(u, delta, A, B, C, D=None, delta_softplus=True)
    got_no_d = selective_scan(mu, md, mA, mB, mC, D=None, delta_softplus=True)
    assert _max_abs_diff(got_no_d, ref_no_d) < 1e-4

    ref_d = selective_scan_np(u, delta, A, B, C, D=D, delta_softplus=True)
    got_d = selective_scan(mu, md, mA, mB, mC, D=mD, delta_softplus=True)
    assert _max_abs_diff(got_d, ref_d) < 1e-4

    # D must actually change the output (sanity on the skip term).
    assert _max_abs_diff(got_d, ref_no_d) > 1e-3


def test_z_gating_via_silu():
    u, delta, A, B, C, D, z, _bias = _make_inputs(5)
    ref = selective_scan_np(u, delta, A, B, C, D=D, z=z, delta_softplus=True)
    mu, md, mA, mB, mC, mD, mz = _to_mx(u, delta, A, B, C, D, z)
    got = selective_scan(mu, md, mA, mB, mC, D=mD, z=mz, delta_softplus=True)
    assert _max_abs_diff(got, ref) < 1e-4


def test_full_path_all_terms():
    # softplus + bias + D + z together, the configuration SEMamba runs.
    u, delta, A, B, C, D, z, delta_bias = _make_inputs(6)
    ref = selective_scan_np(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias, delta_softplus=True,
    )
    mu, md, mA, mB, mC, mD, mz, mbias = _to_mx(
        u, delta, A, B, C, D, z, delta_bias
    )
    got = selective_scan(
        mu, md, mA, mB, mC, D=mD, z=mz,
        delta_bias=mbias, delta_softplus=True,
    )
    assert got.shape == (B_DIM, D_INNER, L)
    assert _max_abs_diff(got, ref) < 1e-4


def test_reverse_scan_matches_flipped_reference():
    # The reverse helper must equal: flip inputs on L, scan, flip output back.
    u, delta, A, B, C, D, z, delta_bias = _make_inputs(7)

    def flip(a):
        return a[:, :, ::-1] if a.ndim == 3 else a

    ref_fwd_on_flipped = selective_scan_np(
        flip(u), flip(delta), A, flip(B), flip(C),
        D=D, z=flip(z), delta_bias=delta_bias, delta_softplus=True,
    )
    ref_reverse = ref_fwd_on_flipped[:, :, ::-1]

    mu, md, mA, mB, mC, mD, mz, mbias = _to_mx(
        u, delta, A, B, C, D, z, delta_bias
    )
    got = selective_scan_reverse(
        mu, md, mA, mB, mC, D=mD, z=mz,
        delta_bias=mbias, delta_softplus=True,
    )
    assert got.shape == (B_DIM, D_INNER, L)
    assert _max_abs_diff(got, ref_reverse) < 1e-4

    # Reverse must differ from forward for a non-symmetric sequence.
    fwd = selective_scan(
        mu, md, mA, mB, mC, D=mD, z=mz,
        delta_bias=mbias, delta_softplus=True,
    )
    assert _max_abs_diff(got, np.asarray(fwd, dtype=np.float64)) > 1e-3


def test_no_softplus_path():
    # delta_softplus=False must skip the softplus and still match upstream.
    # Use positive delta so exp(delta*A) stays < 1 (A < 0) and the recurrence
    # decays; this keeps float32 magnitudes bounded for a tight comparison.
    u, delta, A, B, C, _D, _z, _bias = _make_inputs(8)
    delta = np.abs(delta) + 0.5
    ref = selective_scan_np(u, delta, A, B, C, delta_softplus=False)
    mu, md, mA, mB, mC = _to_mx(u, delta, A, B, C)
    got = selective_scan(mu, md, mA, mB, mC, delta_softplus=False)
    assert _max_abs_diff(got, ref) < 1e-4

    # The softplus branch genuinely changes the output for the same delta.
    got_softplus = selective_scan(mu, md, mA, mB, mC, delta_softplus=True)
    assert _max_abs_diff(got_softplus, ref) > 1e-3


def test_source_imports_without_torch():
    import sys

    import mlx_speech.models.reuse.mamba.scan as scan_mod  # noqa: F401

    assert "torch" not in sys.modules
