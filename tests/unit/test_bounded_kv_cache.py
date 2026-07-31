from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models._cache import BoundedKVCache


def _values(length: int, *, dtype: mx.Dtype, start: int = 0) -> mx.array:
    return (
        mx.arange(start, start + length * 2, dtype=mx.float32)
        .reshape(1, length, 1, 2)
        .astype(dtype)
    )


def test_bounded_cache_appends_into_storage_and_exposes_valid_tuple_prefix() -> None:
    cache = BoundedKVCache.allocate(
        batch_size=1,
        capacity=4,
        num_heads=1,
        head_dim=2,
        key_dtype=mx.float32,
        value_dtype=mx.bfloat16,
    )
    keys = _values(2, dtype=mx.float32)
    values = _values(2, dtype=mx.bfloat16, start=10)

    cache.append(keys, values)
    fetched_keys, fetched_values = cache.fetch()
    iterated_keys, iterated_values = cache
    mx.eval(fetched_keys, fetched_values)

    assert cache.offset == 2
    assert cache.current_length == 2
    assert cache.capacity == 4
    assert cache.keys.shape == (1, 4, 1, 2)
    assert fetched_keys.shape == (1, 2, 1, 2)
    assert fetched_values.shape == (1, 2, 1, 2)
    assert iterated_keys.shape == fetched_keys.shape
    assert iterated_values.shape == fetched_values.shape
    assert cache[0].shape == fetched_keys.shape
    assert cache[1].shape == fetched_values.shape
    assert cache.keys.dtype == mx.float32
    assert cache.values.dtype == mx.bfloat16
    assert mx.array_equal(fetched_keys, keys).item()
    assert mx.array_equal(fetched_values, values).item()

    cache.append(_values(2, dtype=mx.float32, start=20), _values(2, dtype=mx.bfloat16))
    assert cache.offset == 4
    with pytest.raises(ValueError, match="overflow"):
        cache.append(_values(1, dtype=mx.float32), _values(1, dtype=mx.bfloat16))


def test_bounded_cache_grows_in_blocks_without_exceeding_maximum() -> None:
    cache = BoundedKVCache.allocate(
        batch_size=1,
        capacity=256,
        num_heads=1,
        head_dim=2,
        key_dtype=mx.float32,
        value_dtype=mx.bfloat16,
        max_capacity=600,
        growth_step=256,
    )
    first_keys = _values(255, dtype=mx.float32)
    first_values = _values(255, dtype=mx.bfloat16)
    cache.append(first_keys, first_values)
    cache.append(_values(2, dtype=mx.float32), _values(2, dtype=mx.bfloat16))

    assert cache.offset == 257
    assert cache.capacity == 512
    assert cache.keys.dtype == mx.float32
    assert cache.values.dtype == mx.bfloat16
    assert mx.array_equal(cache.fetch()[0][:, :255], first_keys).item()
    assert mx.array_equal(cache.fetch()[1][:, :255], first_values).item()

    cache.append(
        _values(343, dtype=mx.float32),
        _values(343, dtype=mx.bfloat16),
    )
    assert cache.offset == 600
    assert cache.capacity == 600
    with pytest.raises(ValueError, match="maximum=600"):
        cache.append(_values(1, dtype=mx.float32), _values(1, dtype=mx.bfloat16))


def test_bounded_cache_rejects_incompatible_shapes_and_dtypes() -> None:
    cache = BoundedKVCache.allocate(
        batch_size=1,
        capacity=2,
        num_heads=1,
        head_dim=2,
        key_dtype=mx.float32,
        value_dtype=mx.bfloat16,
    )

    with pytest.raises(ValueError, match="shapes must match"):
        cache.append(
            mx.zeros((1, 1, 1, 2)),
            mx.zeros((1, 2, 1, 2), dtype=mx.bfloat16),
        )
    with pytest.raises(ValueError, match="dtypes differ"):
        cache.append(
            mx.zeros((1, 1, 1, 2), dtype=mx.bfloat16),
            mx.zeros((1, 1, 1, 2), dtype=mx.bfloat16),
        )
