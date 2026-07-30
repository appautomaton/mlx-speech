from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_speech.models.dots_tts.latent import LatentIO, LatentStatistics


def test_latent_normalization_matches_official_oracle() -> None:
    fixture = np.load("tests/fixtures/dots_tts/soar/latent_io.npz")
    latent_io = LatentIO(
        LatentStatistics(
            mean=mx.array(fixture["mean"]),
            variance=mx.array(fixture["variance"]),
        )
    )
    normalized = latent_io.normalize(mx.array(fixture["latent"]))
    restored = latent_io.denormalize(normalized)
    mx.eval(normalized, restored)
    np.testing.assert_allclose(normalized, fixture["normalized"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(restored, fixture["restored"], atol=1e-5, rtol=1e-5)


def test_latent_distribution_sampling_uses_injected_noise() -> None:
    latent_io = LatentIO(
        LatentStatistics(mean=mx.zeros((4,)), variance=mx.ones((4,)))
    )
    distribution = mx.concatenate(
        (mx.ones((1, 4, 3)), mx.zeros((1, 4, 3))), axis=1
    )
    sampled = latent_io.sample_distribution(
        distribution, noise=mx.full((1, 4, 3), 0.5)
    )
    mx.eval(sampled)
    assert sampled.shape == (1, 3, 4)
    np.testing.assert_allclose(sampled, 1.5)


def test_latent_statistics_load_strict_safetensors(tmp_path) -> None:
    path = tmp_path / "latent_stats.safetensors"
    save_file(
        {
            "mean": np.zeros(4, dtype=np.float32),
            "var": np.ones(4, dtype=np.float32),
        },
        path,
    )
    statistics = LatentStatistics.from_path(path)
    assert statistics.mean.dtype == mx.float32
    save_file({"mean": np.zeros(4, dtype=np.float32)}, path)
    with pytest.raises(ValueError, match="exactly mean and var"):
        LatentStatistics.from_path(path)


def test_latent_io_rejects_invalid_shapes_and_variance() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        LatentStatistics(mean=mx.zeros((4,)), variance=mx.zeros((4,)))
    latent_io = LatentIO(
        LatentStatistics(mean=mx.zeros((4,)), variance=mx.ones((4,)))
    )
    with pytest.raises(ValueError, match="channel count"):
        latent_io.sample_distribution(mx.zeros((1, 6, 3)))
    with pytest.raises(ValueError, match="shape"):
        latent_io.normalize(mx.zeros((1, 3, 5)))
