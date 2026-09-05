"""Numerical helpers for FireRedTTS3 Base latent generation."""

from __future__ import annotations

import math

import mlx.core as mx


def cosine_time_schedule(steps: int) -> mx.array:
    if steps <= 0:
        raise ValueError("FireRedTTS3 flow steps must be positive")
    linear = mx.linspace(0.0, 1.0, steps + 1, dtype=mx.float32)
    return 1.0 - mx.cos(linear * (0.5 * math.pi))


def classifier_free_guidance(
    conditional: mx.array,
    unconditional: mx.array,
    scale: float,
) -> mx.array:
    return (1.0 + scale) * conditional - scale * unconditional


def euler_flow_step(
    state: mx.array,
    velocity: mx.array,
    delta: mx.array | float,
) -> mx.array:
    return state + delta * velocity


__all__ = ["classifier_free_guidance", "cosine_time_schedule", "euler_flow_step"]
