"""SOAR and MeanFlow next-patch solvers for dots.tts."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ODESchedule:
    """Fixed Euler schedule shared by cached and full-history solvers."""

    mode: str
    times: mx.array
    step_size: float


def build_ode_schedule(mode: str, steps: int | None, dtype: mx.Dtype) -> ODESchedule:
    resolved = resolve_solver_steps(mode, steps)
    step_size = 1.0 / resolved
    return ODESchedule(
        mode=mode,
        times=mx.array([index * step_size for index in range(resolved)], dtype=dtype),
        step_size=step_size,
    )


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


def _expand_cfg_mask(mask: mx.array | None, batch_size: int) -> mx.array | None:
    if mask is None or mask.ndim == 2:
        return mask
    if mask.ndim not in {3, 4} or mask.shape[0] not in {1, batch_size}:
        raise ValueError("SOAR attention mask batch must be one or match the input")
    if mask.shape[0] == batch_size:
        return mx.concatenate((mask, mask), axis=0)
    return mask


def _expand_cfg_positions(
    positions: mx.array | None, batch_size: int
) -> mx.array | None:
    if positions is None or positions.ndim == 1:
        return positions
    if positions.ndim != 2 or positions.shape[0] not in {1, batch_size}:
        raise ValueError("SOAR position batch must be one or match the input")
    if positions.shape[0] == batch_size:
        return mx.concatenate((positions, positions), axis=0)
    return positions


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
            attention_mask=_expand_cfg_mask(attention_mask, batch_size),
            positions=_expand_cfg_positions(positions, batch_size),
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
        schedule = build_ode_schedule("soar", steps, sequence.dtype)
        batch_size = int(sequence.shape[0])
        expected = (batch_size, int(patch_size), self.latent_dim)
        coordinate = (
            mx.random.normal(expected).astype(sequence.dtype)
            if noise is None
            else noise.astype(sequence.dtype)
        )
        if coordinate.shape != expected:
            raise ValueError(f"SOAR noise must have shape {expected}")
        for index in range(int(schedule.times.shape[0])):
            timestep = schedule.times[index : index + 1]
            coordinate = coordinate + schedule.step_size * self.velocity(
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
        schedule = build_ode_schedule("meanflow", steps, sequence.dtype)
        batch_size = int(sequence.shape[0])
        expected = (batch_size, int(patch_size), self.latent_dim)
        coordinate = (
            mx.random.normal(expected).astype(sequence.dtype)
            if noise is None
            else noise.astype(sequence.dtype)
        )
        if coordinate.shape != expected:
            raise ValueError(f"MeanFlow noise must have shape {expected}")
        for index in range(int(schedule.times.shape[0])):
            conditioned, latent_start = splice_coordinate(
                sequence, coordinate, self.coordinate_projection
            )
            timestep = mx.broadcast_to(schedule.times[index : index + 1], (batch_size,))
            duration = mx.full((batch_size,), schedule.step_size, dtype=sequence.dtype)
            velocity = self.dit(
                conditioned,
                timestep,
                duration=duration,
                attention_mask=attention_mask,
                positions=positions,
                speaker_condition=speaker_condition,
            )[:, latent_start:]
            coordinate = coordinate + velocity * schedule.step_size
        return coordinate


__all__ = [
    "MeanFlowSolver",
    "ODESchedule",
    "SOARSolver",
    "build_ode_schedule",
    "resolve_solver_steps",
    "splice_coordinate",
]
