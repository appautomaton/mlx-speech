import mlx.core as mx
from typing import Optional, Tuple


class KVCache:
    """KV Cache for autoregressive generation.

    Stores key/value states for efficient inference.
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

        self._keys: Optional[mx.array] = None
        self._values: Optional[mx.array] = None

    @property
    def offset(self) -> int:
        return max(self._offsets)

    def update(self, layer_idx: int, key: mx.array, value: mx.array):
        """Update cache with new key/value.

        Args:
            layer_idx: layer index
            key: (batch, heads, seq, head_dim)
            value: (batch, heads, seq, head_dim)
        """
        seq_len = key.shape[2]

        if self._keys is None:
            batch = key.shape[0]
            heads = key.shape[1]
            head_dim = key.shape[3]
            self._keys = mx.zeros(
                (self.num_layers, batch, heads, self.max_length, head_dim)
            )
            self._values = mx.zeros(
                (self.num_layers, batch, heads, self.max_length, head_dim)
            )

        start = self._offsets[layer_idx]
        end = start + seq_len
        self._keys[layer_idx, :, :, start:end] = key
        self._values[layer_idx, :, :, start:end] = value
        self._offsets[layer_idx] = end

    def get(self, layer_idx: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """Get cached keys/values.

        Args:
            layer_idx: specific layer, or None for all

        Returns:
            (keys, values) - sliced to current offset
        """
        if layer_idx is not None:
            return (
                self._keys[layer_idx, :, :, : self._offsets[layer_idx]],
                self._values[layer_idx, :, :, : self._offsets[layer_idx]],
            )
        keys_out = []
        vals_out = []
        for i in range(self.num_layers):
            keys_out.append(self._keys[i, :, :, : self._offsets[i]])
            vals_out.append(self._values[i, :, :, : self._offsets[i]])
        return keys_out, vals_out

    def reset(self):
        """Reset cache for new generation."""
        self._offsets = [0] * self.num_layers

    def trim_to(self, length: int):
        """Trim cache to specific length."""
        for i in range(self.num_layers):
            self._offsets[i] = min(length, self._offsets[i])
