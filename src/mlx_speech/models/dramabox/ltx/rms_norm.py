"""Functional RMSNorm — the upstream LTX code uses `torch.nn.functional`-style
RMSNorm with no learnable weight inside the connector blocks (only the
attention's q_norm/k_norm have weights). This file exposes the same
"functional" form.

The DiT's per-head attention RMSNorm and the connector's full-inner-dim
RMSNorm both use this when there is no weight; with a weight, it falls
through to the standard formula.
"""

from __future__ import annotations

import mlx.core as mx


def rms_norm(x: mx.array, eps: float = 1e-6, weight: mx.array | None = None) -> mx.array:
    """Standard RMSNorm: ``x * rsqrt(mean(x**2) + eps) * weight`` (weight=1 if None).

    Norm is computed in fp32 for stability, then cast back. This is the same
    contract as the upstream `ltx_core.utils.rms_norm` used by
    `_BasicTransformerBlock1D` for its functional pre-norms.
    """
    orig_dtype = x.dtype
    x32 = x.astype(mx.float32)
    var = mx.mean(x32 * x32, axis=-1, keepdims=True)
    out = x32 * mx.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.astype(mx.float32)
    return out.astype(orig_dtype)


__all__ = ["rms_norm"]
