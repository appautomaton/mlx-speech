from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx

# Buffers grow in fixed-size chunks so most decode steps write into
# already-allocated storage instead of reallocating.
_ALLOC_STEP = 256


class KVCache:
    """Per-layer KV cache with chunked buffer growth.

    Each layer owns its own ``(batch, heads, capacity, head_dim)`` buffer, so
    a slice update touches only that layer's storage and MLX can donate the
    buffer and write in place. Storing every layer in one shared
    ``(num_layers, ...)`` tensor forces a full-tensor copy per update — at
    36 layers x (K+V) per generated token that is multi-GB of memcopy per
    token and made decode time grow with memory pressure.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        max_length: int = 8192,
    ):
        self.num_layers = num_layers
        self.dim = dim
        self.max_length = max_length
        self._offsets = [0] * num_layers

        self._keys: List[Optional[mx.array]] = [None] * num_layers
        self._values: List[Optional[mx.array]] = [None] * num_layers

    @property
    def offset(self) -> int:
        return max(self._offsets)

    @property
    def current_length(self) -> int:
        return self.offset

    def update(self, layer_idx: int, key: mx.array, value: mx.array):
        """Update cache with new key/value.

        Args:
            layer_idx: layer index
            key: (batch, heads, seq, head_dim)
            value: (batch, heads, seq, head_dim)
        """
        seq_len = key.shape[2]
        start = self._offsets[layer_idx]
        end = start + seq_len

        keys = self._keys[layer_idx]
        if keys is None or end > keys.shape[2]:
            batch, heads, _, head_dim = key.shape
            capacity = min(
                ((end + _ALLOC_STEP - 1) // _ALLOC_STEP) * _ALLOC_STEP,
                max(self.max_length, end),
            )
            new_keys = mx.zeros(
                (batch, heads, capacity, head_dim), dtype=key.dtype
            )
            new_values = mx.zeros(
                (batch, heads, capacity, head_dim), dtype=value.dtype
            )
            if keys is not None and start > 0:
                new_keys[..., :start, :] = keys[..., :start, :]
                new_values[..., :start, :] = self._values[layer_idx][
                    ..., :start, :
                ]
            self._keys[layer_idx] = new_keys
            self._values[layer_idx] = new_values

        self._keys[layer_idx][..., start:end, :] = key
        self._values[layer_idx][..., start:end, :] = value
        self._offsets[layer_idx] = end

    def get(self, layer_idx: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """Get cached keys/values.

        Args:
            layer_idx: specific layer, or None for all

        Returns:
            (keys, values) - sliced to current offset
        """
        if layer_idx is not None:
            keys = self._keys[layer_idx]
            values = self._values[layer_idx]
            if keys is None or values is None:
                raise RuntimeError("KV cache is uninitialized")
            offset = self._offsets[layer_idx]
            return (
                keys[..., :offset, :],
                values[..., :offset, :],
            )

        if any(k is None for k in self._keys) or any(
            v is None for v in self._values
        ):
            raise RuntimeError("KV cache is uninitialized")
        offset = self.offset
        return (
            mx.stack([k[..., :offset, :] for k in self._keys], axis=0),
            mx.stack([v[..., :offset, :] for v in self._values], axis=0),
        )

    def reset(self):
        """Reset logical offsets while keeping allocated storage."""
        self._offsets = [0] * self.num_layers

    def trim_to(self, length: int):
        """Trim cache to specific length."""
        for i in range(self.num_layers):
            self._offsets[i] = min(length, self._offsets[i])
