"""KV cache helpers for MossTTSLocal inference."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .config import Qwen3LanguageConfig


@dataclass
class _BaseLayerKVCache:
    batch_size: int
    num_kv_heads: int
    max_length: int
    head_dim: int
    dtype: mx.Dtype

    def __post_init__(self) -> None:
        self.keys = mx.zeros(
            (self.batch_size, self.num_kv_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.values = mx.zeros(
            (self.batch_size, self.num_kv_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.current_length = 0

    def reset(self) -> None:
        self.current_length = 0

    def append(self, key_states: mx.array, value_states: mx.array) -> None:
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Expected key/value shapes to match, got {key_states.shape} vs {value_states.shape}."
            )
        if key_states.ndim != 4:
            raise ValueError(
                "Expected key/value with shape (batch, num_kv_heads, seq, head_dim), "
                f"got {key_states.shape}."
            )
        step = int(key_states.shape[2])
        end = self.current_length + step
        if end > self.max_length:
            raise ValueError(
                f"KV cache overflow: need {end} slots, cache only has {self.max_length}."
            )
        self.keys[:, :, self.current_length:end, :] = key_states
        self.values[:, :, self.current_length:end, :] = value_states
        self.current_length = end

    def get(self) -> tuple[mx.array, mx.array]:
        return (
            self.keys[:, :, : self.current_length, :],
            self.values[:, :, : self.current_length, :],
        )


@dataclass
class GlobalLayerKVCache(_BaseLayerKVCache):
    """Per-layer global transformer KV cache."""


@dataclass
class LocalLayerKVCache(_BaseLayerKVCache):
    """Per-layer local transformer KV cache."""


@dataclass
class GlobalKVCache:
    """Per-layer cache for the outer time-axis autoregressive decode."""

    layers: tuple[GlobalLayerKVCache, ...]
    prompt_length: int = 0

    @property
    def current_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0].current_length)

    @classmethod
    def allocate(
        cls,
        config: Qwen3LanguageConfig,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype,
    ) -> "GlobalKVCache":
        layers = tuple(
            GlobalLayerKVCache(
                batch_size=batch_size,
                num_kv_heads=config.num_key_value_heads,
                max_length=max_length,
                head_dim=config.effective_head_dim,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )
        return cls(layers=layers)


@dataclass
class LocalKVCache:
    """Per-layer cache for the inner RVQ depth autoregressive decode."""

    layers: tuple[LocalLayerKVCache, ...]

    @property
    def current_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0].current_length)

    @classmethod
    def allocate(
        cls,
        config: Qwen3LanguageConfig,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype,
    ) -> "LocalKVCache":
        layers = tuple(
            LocalLayerKVCache(
                batch_size=batch_size,
                num_kv_heads=config.num_key_value_heads,
                max_length=max_length,
                head_dim=config.effective_head_dim,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )
        return cls(layers=layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
