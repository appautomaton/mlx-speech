"""Internal capacity-managed key/value cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import mlx.core as mx


@dataclass
class BoundedKVCache:
    """Append-only K/V storage with a valid-prefix tuple view."""

    keys: mx.array
    values: mx.array
    offset: int = 0
    max_capacity: int | None = None
    growth_step: int | None = None

    def __post_init__(self) -> None:
        self._validate_storage()
        if self.max_capacity is None:
            self.max_capacity = self.capacity
        if self.max_capacity < self.capacity:
            raise ValueError(
                "bounded K/V cache maximum is smaller than its capacity: "
                f"maximum={self.max_capacity} capacity={self.capacity}"
            )
        if self.growth_step is not None and self.growth_step <= 0:
            raise ValueError("bounded K/V cache growth_step must be positive")

    def _validate_storage(self) -> None:
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise ValueError(
                "bounded K/V cache arrays must have shape (batch, sequence, heads, dim)"
            )
        if self.keys.shape != self.values.shape:
            raise ValueError(
                "bounded K/V cache key/value shapes must match: "
                f"{self.keys.shape} vs {self.values.shape}"
            )
        if self.capacity <= 0:
            raise ValueError("bounded K/V cache capacity must be positive")
        if self.offset < 0 or self.offset > self.capacity:
            raise ValueError(
                "bounded K/V cache offset is outside its capacity: "
                f"offset={self.offset} capacity={self.capacity}"
            )
        if self.max_capacity is not None and self.max_capacity < self.capacity:
            raise ValueError(
                "bounded K/V cache maximum is smaller than its capacity: "
                f"maximum={self.max_capacity} capacity={self.capacity}"
            )

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[1])

    @property
    def current_length(self) -> int:
        return self.offset

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        capacity: int,
        num_heads: int,
        head_dim: int,
        key_dtype: mx.Dtype,
        value_dtype: mx.Dtype,
        max_capacity: int | None = None,
        growth_step: int | None = None,
    ) -> "BoundedKVCache":
        if min(batch_size, capacity, num_heads, head_dim) <= 0:
            raise ValueError("bounded K/V cache dimensions must be positive")
        shape = (batch_size, capacity, num_heads, head_dim)
        return cls(
            keys=mx.zeros(shape, dtype=key_dtype),
            values=mx.zeros(shape, dtype=value_dtype),
            max_capacity=max_capacity,
            growth_step=growth_step,
        )

    @classmethod
    def from_values(
        cls,
        keys: mx.array,
        values: mx.array,
        *,
        capacity: int,
        max_capacity: int | None = None,
        growth_step: int | None = None,
    ) -> "BoundedKVCache":
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "bounded K/V cache values must have shape (batch, sequence, heads, dim)"
            )
        if keys.shape != values.shape:
            raise ValueError(
                "bounded K/V cache key/value shapes must match: "
                f"{keys.shape} vs {values.shape}"
            )
        cache = cls.allocate(
            batch_size=int(keys.shape[0]),
            capacity=capacity,
            num_heads=int(keys.shape[2]),
            head_dim=int(keys.shape[3]),
            key_dtype=keys.dtype,
            value_dtype=values.dtype,
            max_capacity=max_capacity,
            growth_step=growth_step,
        )
        cache.append(keys, values)
        return cache

    def _validate_append(self, keys: mx.array, values: mx.array) -> None:
        self._validate_storage()
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "bounded K/V cache append expects (batch, sequence, heads, dim) arrays"
            )
        if keys.shape != values.shape:
            raise ValueError(
                "bounded K/V cache append key/value shapes must match: "
                f"{keys.shape} vs {values.shape}"
            )
        expected = (self.keys.shape[0], *self.keys.shape[2:])
        actual = (keys.shape[0], *keys.shape[2:])
        if actual != expected:
            raise ValueError(
                "bounded K/V cache append shape is incompatible with its storage: "
                f"{keys.shape} vs {self.keys.shape}"
            )
        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise ValueError(
                "bounded K/V cache append dtypes differ from its storage: "
                f"keys {keys.dtype} vs {self.keys.dtype}, "
                f"values {values.dtype} vs {self.values.dtype}"
            )

    def validate_append_length(self, length: int) -> None:
        """Validate capacity for an append without changing cache state."""

        self._validate_storage()
        if length < 0:
            raise ValueError("bounded K/V cache append length must be non-negative")
        required = self.offset + length
        if required <= self.capacity:
            return
        maximum = int(self.max_capacity)
        if required > maximum or self.growth_step is None:
            raise ValueError(
                "bounded K/V cache overflow: "
                f"required={required} capacity={self.capacity} maximum={maximum}"
            )

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.capacity:
            return
        maximum = int(self.max_capacity)
        target = min(
            ((required + self.growth_step - 1) // self.growth_step) * self.growth_step,
            maximum,
        )
        if target < required:
            raise ValueError(
                "bounded K/V cache overflow: "
                f"required={required} capacity={self.capacity} maximum={maximum}"
            )
        shape = (self.keys.shape[0], target, *self.keys.shape[2:])
        next_keys = mx.zeros(shape, dtype=self.keys.dtype)
        next_values = mx.zeros(shape, dtype=self.values.dtype)
        if self.offset:
            next_keys[:, : self.offset, :, :] = self.keys[:, : self.offset, :, :]
            next_values[:, : self.offset, :, :] = self.values[:, : self.offset, :, :]
        self.keys = next_keys
        self.values = next_values

    def append(self, keys: mx.array, values: mx.array) -> None:
        self._validate_append(keys, values)
        length = int(keys.shape[1])
        self.validate_append_length(length)
        end = self.offset + length
        self._ensure_capacity(end)
        self.keys[:, self.offset : end, :, :] = keys
        self.values[:, self.offset : end, :, :] = values
        self.offset = end

    def restore_offset(self, offset: int) -> None:
        """Restore a prior valid offset after a failed multi-cache operation."""

        if offset < 0 or offset > self.offset:
            raise ValueError(
                "bounded K/V cache restore offset is invalid: "
                f"offset={offset} current={self.offset}"
            )
        self.offset = offset

    def fetch(self) -> tuple[mx.array, mx.array]:
        """Return key and value views restricted to the valid prefix."""

        return (
            self.keys[:, : self.offset, :, :],
            self.values[:, : self.offset, :, :],
        )

    def __iter__(self) -> Iterator[mx.array]:
        return iter(self.fetch())

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int | slice) -> mx.array | tuple[mx.array, ...]:
        return self.fetch()[index]


__all__ = ["BoundedKVCache"]
