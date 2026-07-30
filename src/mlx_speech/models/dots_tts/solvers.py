"""SOAR and MeanFlow next-patch solvers for dots.tts."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx
import mlx.nn as nn


class _DiTPredictor(Protocol):
    hidden_size: int

    def __call__(
        self,
        sequence: mx.array,
        timesteps: mx.array,
        *,
        duration: mx.array | None = None,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        speaker_condition: mx.array | None = None,
    ) -> mx.array: ...


def resolve_solver_steps(mode: str, steps: int | None) -> int:
    if mode not in {"soar", "meanflow"}:
        raise ValueError(f"unsupported dots.tts solver mode: {mode}")
    resolved = (4 if mode == "meanflow" else 10) if steps is None else int(steps)
    if resolved <= 0:
        raise ValueError("solver steps must be positive")
    return resolved


def splice_coordinate(
    sequence: mx.array,
    coordinate: mx.array,
    projection: nn.Module,
) -> tuple[mx.array, int]:
    """Project a noisy latent patch and replace the reserved tail slots."""

    if sequence.ndim != 3 or coordinate.ndim != 3:
        raise ValueError("solver sequences and coordinates must be rank three")
    if int(sequence.shape[0]) != int(coordinate.shape[0]):
        raise ValueError("solver sequence and coordinate batch sizes differ")
    patch_size = int(coordinate.shape[1])
    if patch_size <= 0 or int(sequence.shape[1]) < patch_size:
        raise ValueError("solver sequence does not reserve a latent patch tail")
    projected = projection(coordinate).astype(sequence.dtype)
    if projected.shape != (
        sequence.shape[0],
        patch_size,
        sequence.shape[-1],
    ):
        raise ValueError("coordinate projection does not match sequence hidden size")
    latent_start = int(sequence.shape[1]) - patch_size
    return mx.concatenate((sequence[:, :latent_start], projected), axis=1), latent_start


class SOARSolver:
    """Fixed-step Euler flow matching with batched classifier-free guidance."""

    def __init__(
        self,
        dit: _DiTPredictor,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
    ):
        self.dit = dit
        self.coordinate_projection = coordinate_projection
        self.latent_dim = int(latent_dim)

    def velocity(
        self,
        coordinate: mx.array,
        timestep: mx.array,
        *,
        sequence: mx.array,
        cfg_sequence: mx.array,
        attention_mask: mx.array | None,
        positions: mx.array | None,
        speaker_condition: mx.array | None,
        guidance_scale: float,
    ) -> mx.array:
        if sequence.shape != cfg_sequence.shape:
            raise ValueError("SOAR conditional and CFG sequences must have equal shape")
        conditional, latent_start = splice_coordinate(
            sequence, coordinate, self.coordinate_projection
        )
        unconditional, _ = splice_coordinate(
            cfg_sequence, coordinate, self.coordinate_projection
        )
        branches = mx.concatenate((conditional, unconditional), axis=0)
        batch_size = int(sequence.shape[0])
        times = mx.broadcast_to(timestep.reshape(1), (2 * batch_size,))
        speaker_branches = None
        if speaker_condition is not None:
            speaker_branches = mx.concatenate(
                (speaker_condition, mx.zeros_like(speaker_condition)), axis=0
            )
        prediction = self.dit(
            branches,
            times,
            attention_mask=attention_mask,
            positions=positions,
            speaker_condition=speaker_branches,
        )[:, latent_start:]
        conditional_velocity = prediction[:batch_size]
        unconditional_velocity = prediction[batch_size:]
        return conditional_velocity + float(guidance_scale) * (
            conditional_velocity - unconditional_velocity
        )

    def sample(
        self,
        *,
        sequence: mx.array,
        cfg_sequence: mx.array,
        attention_mask: mx.array | None,
        positions: mx.array | None,
        speaker_condition: mx.array | None,
        guidance_scale: float = 1.2,
        steps: int | None = None,
        patch_size: int = 4,
        noise: mx.array | None = None,
    ) -> mx.array:
        steps = resolve_solver_steps("soar", steps)
        batch_size = int(sequence.shape[0])
        expected = (batch_size, int(patch_size), self.latent_dim)
        coordinate = (
            mx.random.normal(expected).astype(sequence.dtype)
            if noise is None
            else noise.astype(sequence.dtype)
        )
        if coordinate.shape != expected:
            raise ValueError(f"SOAR noise must have shape {expected}")
        step_size = 1.0 / steps
        for index in range(steps):
            timestep = mx.array([index * step_size], dtype=sequence.dtype)
            coordinate = coordinate + step_size * self.velocity(
                coordinate,
                timestep,
                sequence=sequence,
                cfg_sequence=cfg_sequence,
                attention_mask=attention_mask,
                positions=positions,
                speaker_condition=speaker_condition,
                guidance_scale=guidance_scale,
            )
        return coordinate


class MeanFlowSolver:
    """Four-evaluation distilled MeanFlow sampler with no runtime CFG."""

    def __init__(
        self,
        dit: _DiTPredictor,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
    ):
        self.dit = dit
        self.coordinate_projection = coordinate_projection
        self.latent_dim = int(latent_dim)

    def sample(
        self,
        *,
        sequence: mx.array,
        attention_mask: mx.array | None,
        positions: mx.array | None,
        speaker_condition: mx.array | None,
        steps: int | None = None,
        patch_size: int = 4,
        noise: mx.array | None = None,
    ) -> mx.array:
        steps = resolve_solver_steps("meanflow", steps)
        batch_size = int(sequence.shape[0])
        expected = (batch_size, int(patch_size), self.latent_dim)
        coordinate = (
            mx.random.normal(expected).astype(sequence.dtype)
            if noise is None
            else noise.astype(sequence.dtype)
        )
        if coordinate.shape != expected:
            raise ValueError(f"MeanFlow noise must have shape {expected}")
        step_size = 1.0 / steps
        for index in range(steps):
            conditioned, latent_start = splice_coordinate(
                sequence, coordinate, self.coordinate_projection
            )
            timestep = mx.full((batch_size,), index * step_size, dtype=sequence.dtype)
            duration = mx.full((batch_size,), step_size, dtype=sequence.dtype)
            velocity = self.dit(
                conditioned,
                timestep,
                duration=duration,
                attention_mask=attention_mask,
                positions=positions,
                speaker_condition=speaker_condition,
            )[:, latent_start:]
            coordinate = coordinate + velocity * step_size
        return coordinate


__all__ = [
    "MeanFlowSolver",
    "SOARSolver",
    "resolve_solver_steps",
    "splice_coordinate",
]
