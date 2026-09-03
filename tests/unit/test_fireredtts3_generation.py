from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from mlx_speech.generation.fireredtts3 import (
    classifier_free_guidance,
    cosine_time_schedule,
    euler_flow_step,
)


def test_cosine_schedule_matches_official_equation() -> None:
    schedule = cosine_time_schedule(2)
    np.testing.assert_allclose(
        schedule,
        [0.0, 1.0 - math.cos(math.pi / 4), 1.0],
        atol=1e-7,
        rtol=0.0,
    )


def test_cfg_and_euler_updates_match_official_equations() -> None:
    conditional = mx.array([[[2.0, 4.0]]])
    unconditional = mx.array([[[1.0, -2.0]]])
    guided = classifier_free_guidance(conditional, unconditional, 2.0)
    updated = euler_flow_step(mx.array([[[0.5, 1.0]]]), guided, 0.25)
    np.testing.assert_allclose(guided, [[[4.0, 16.0]]], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(updated, [[[1.5, 5.0]]], atol=0.0, rtol=0.0)
