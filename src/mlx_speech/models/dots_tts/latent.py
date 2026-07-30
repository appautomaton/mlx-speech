"""Torch-free dots.tts latent sampling and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class LatentStatistics:
    mean: mx.array
    variance: mx.array

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.variance.ndim != 1:
            raise ValueError("latent mean and variance must be rank-1 arrays")
        if self.mean.shape != self.variance.shape:
            raise ValueError("latent mean and variance shapes differ")
        if int(self.mean.shape[0]) <= 0:
            raise ValueError("latent statistics must not be empty")
        mx.eval(self.mean, self.variance)
        if not bool(mx.all(mx.isfinite(self.mean)).item()):
            raise ValueError("latent mean contains non-finite values")
        if not bool(mx.all(mx.isfinite(self.variance)).item()):
            raise ValueError("latent variance contains non-finite values")
        if not bool(mx.all(self.variance > 0).item()):
            raise ValueError("latent variance must be strictly positive")

    @classmethod
    def from_path(cls, path: str | Path) -> "LatentStatistics":
        source = Path(path)
        if source.is_dir():
            source = source / "latent_stats.safetensors"
        if not source.is_file():
            raise FileNotFoundError(f"dots.tts latent statistics not found: {source}")
        try:
            payload = mx.load(str(source))
        except Exception as error:
            raise ValueError(f"invalid dots.tts latent statistics: {source}") from error
        if not isinstance(payload, dict) or set(payload) != {"mean", "var"}:
            raise ValueError("latent_stats.safetensors must contain exactly mean and var")
        return cls(
            mean=payload["mean"].astype(mx.float32),
            variance=payload["var"].astype(mx.float32),
        )


class LatentIO:
    """Normalize continuous latents and sample AudioVAE distributions."""

    def __init__(self, statistics: LatentStatistics):
        self.statistics = statistics
        self.standard_deviation = mx.sqrt(statistics.variance)

    @property
    def latent_dim(self) -> int:
        return int(self.statistics.mean.shape[0])

    def _validate_last_dim(self, value: mx.array, name: str) -> None:
        if value.ndim != 3 or int(value.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"{name} must have shape (batch, time, {self.latent_dim}), "
                f"got {value.shape}"
            )

    def normalize(self, latent: mx.array) -> mx.array:
        self._validate_last_dim(latent, "latent")
        return (latent - self.statistics.mean) / self.standard_deviation

    def denormalize(self, normalized: mx.array) -> mx.array:
        self._validate_last_dim(normalized, "normalized latent")
        return normalized * self.standard_deviation + self.statistics.mean

    def sample_distribution(
        self,
        distribution: mx.array,
        *,
        noise: mx.array | None = None,
    ) -> mx.array:
        if distribution.ndim != 3:
            raise ValueError(
                "AudioVAE distribution must have shape (batch, 2*latent_dim, time)"
            )
        if int(distribution.shape[1]) != 2 * self.latent_dim:
            raise ValueError(
                f"AudioVAE distribution channel count must be {2 * self.latent_dim}, "
                f"got {distribution.shape[1]}"
            )
        mean, log_standard_deviation = mx.split(distribution, 2, axis=1)
        if noise is None:
            noise = mx.random.normal(mean.shape)
        elif noise.shape != mean.shape:
            raise ValueError(
                f"latent noise shape {noise.shape} does not match mean {mean.shape}"
            )
        sampled = mean + noise * mx.exp(log_standard_deviation)
        return sampled.transpose(0, 2, 1)


__all__ = ["LatentIO", "LatentStatistics"]
