from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_speech.models.dots_tts.solvers import (
    MeanFlowSolver,
    SOARSolver,
    resolve_solver_steps,
    splice_coordinate,
)


def _projection(latent_dim: int, hidden_size: int) -> nn.Linear:
    projection = nn.Linear(latent_dim, hidden_size, bias=True)
    projection.weight = mx.zeros((hidden_size, latent_dim))
    projection.bias = mx.zeros((hidden_size,))
    return projection


class _SOARPredictor:
    hidden_size = 8

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        sequence: mx.array,
        timesteps: mx.array,
        *,
        duration: mx.array | None = None,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        speaker_condition: mx.array | None = None,
    ) -> mx.array:
        self.calls.append(
            {
                "batch": int(sequence.shape[0]),
                "times": np.asarray(timesteps).tolist(),
                "duration": duration,
                "speaker": speaker_condition,
            }
        )
        branch_value = sequence[:, :1, :1]
        return mx.broadcast_to(
            branch_value, (sequence.shape[0], sequence.shape[1], self.latent_dim)
        )


class _MeanFlowPredictor:
    hidden_size = 8

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self.calls: list[tuple[int, float, float]] = []

    def __call__(
        self,
        sequence: mx.array,
        timesteps: mx.array,
        *,
        duration: mx.array | None = None,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        speaker_condition: mx.array | None = None,
    ) -> mx.array:
        assert duration is not None
        self.calls.append(
            (
                int(sequence.shape[0]),
                float(timesteps[0].item()),
                float(duration[0].item()),
            )
        )
        return mx.full(
            (sequence.shape[0], sequence.shape[1], self.latent_dim), 0.1
        )


def test_splice_coordinate_preserves_prefix_and_replaces_tail() -> None:
    projection = nn.Linear(2, 4, bias=True)
    projection.weight = mx.ones((4, 2))
    projection.bias = mx.zeros((4,))
    sequence = mx.arange(24, dtype=mx.float32).reshape(1, 6, 4)
    coordinate = mx.ones((1, 2, 2))
    spliced, latent_start = splice_coordinate(sequence, coordinate, projection)
    mx.eval(spliced)
    assert latent_start == 4
    np.testing.assert_array_equal(spliced[:, :4], sequence[:, :4])
    np.testing.assert_array_equal(spliced[:, 4:], mx.full((1, 2, 4), 2.0))


def test_soar_uses_euler_cfg_and_approved_defaults() -> None:
    latent_dim = 3
    predictor = _SOARPredictor(latent_dim)
    solver = SOARSolver(
        predictor, _projection(latent_dim, 8), latent_dim=latent_dim
    )
    result = solver.sample(
        sequence=mx.ones((1, 6, 8)),
        cfg_sequence=mx.zeros((1, 6, 8)),
        attention_mask=None,
        positions=None,
        speaker_condition=mx.ones((1, 8)),
        noise=mx.zeros((1, 2, latent_dim)),
        patch_size=2,
    )
    mx.eval(result)
    assert len(predictor.calls) == 10
    assert all(call["batch"] == 2 for call in predictor.calls)
    first_speaker = predictor.calls[0]["speaker"]
    assert first_speaker is not None
    np.testing.assert_array_equal(first_speaker[1], mx.zeros((8,)))
    np.testing.assert_allclose(result, 2.2, atol=1e-6)


def test_meanflow_defaults_to_four_evaluations_without_cfg_branch() -> None:
    latent_dim = 3
    predictor = _MeanFlowPredictor(latent_dim)
    solver = MeanFlowSolver(
        predictor, _projection(latent_dim, 8), latent_dim=latent_dim
    )
    result = solver.sample(
        sequence=mx.ones((1, 6, 8)),
        attention_mask=None,
        positions=None,
        speaker_condition=mx.ones((1, 8)),
        noise=mx.zeros((1, 2, latent_dim)),
        patch_size=2,
    )
    mx.eval(result)
    assert [round(call[1], 6) for call in predictor.calls] == [0.0, 0.25, 0.5, 0.75]
    assert all(call[0] == 1 and round(call[2], 6) == 0.25 for call in predictor.calls)
    np.testing.assert_allclose(result, 0.1, atol=1e-6)


def test_solver_noise_is_repeatable_for_a_fixed_mlx_seed() -> None:
    latent_dim = 2
    solver = MeanFlowSolver(
        _MeanFlowPredictor(latent_dim),
        _projection(latent_dim, 8),
        latent_dim=latent_dim,
    )
    kwargs = {
        "sequence": mx.ones((1, 5, 8)),
        "attention_mask": None,
        "positions": None,
        "speaker_condition": None,
        "patch_size": 2,
    }
    mx.random.seed(67)
    first = solver.sample(**kwargs)
    mx.random.seed(67)
    second = solver.sample(**kwargs)
    mx.eval(first, second)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_solver_step_defaults_and_validation() -> None:
    assert resolve_solver_steps("soar", None) == 10
    assert resolve_solver_steps("meanflow", None) == 4
    assert resolve_solver_steps("meanflow", 2) == 2
    with pytest.raises(ValueError, match="positive"):
        resolve_solver_steps("soar", 0)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_solver_steps("unknown", None)
