"""Sampling helpers for LongCat AudioDiT."""

from __future__ import annotations

import mlx.core as mx


def odeint_euler(fn, y0: mx.array, t: mx.array) -> mx.array:
    ys = [y0]
    y = y0
    for index in range(int(t.shape[0]) - 1):
        dt = t[index + 1] - t[index]
        y = y + (fn(t[index], y) * dt)
        ys.append(y)
    return mx.stack(ys, axis=0)


class _MomentumBuffer:
    def __init__(self, momentum: float = -0.75) -> None:
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: mx.array) -> None:
        self.running_average = update_value + (self.running_average * self.momentum)


def _project(
    v0: mx.array, v1: mx.array, dims: tuple[int, ...] = (-1, -2)
) -> tuple[mx.array, mx.array]:
    denom = mx.sqrt(mx.sum(mx.square(v1), axis=dims, keepdims=True)) + 1e-6
    v1_normalized = v1 / denom
    v0_parallel = mx.sum(v0 * v1_normalized, axis=dims, keepdims=True) * v1_normalized
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel, v0_orthogonal


def apg_forward(
    pred_cond: mx.array,
    pred_uncond: mx.array,
    *,
    guidance_scale: float,
    momentum_buffer: _MomentumBuffer | None = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims: tuple[int, ...] = (-1, -2),
) -> mx.array:
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0.0:
        diff_norm = mx.sqrt(mx.sum(mx.square(diff), axis=dims, keepdims=True))
        scale_factor = mx.minimum(
            mx.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-6)
        )
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims=dims)
    normalized_update = diff_orthogonal + (eta * diff_parallel)
    return pred_cond + (guidance_scale * normalized_update)
