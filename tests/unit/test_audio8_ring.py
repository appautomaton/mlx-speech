from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_speech.models.audio8_asr_infinite.audio_encoder import (
    Audio8AudioCache,
    Audio8LayerCache,
)
from mlx_speech.models.audio8_asr_infinite.cache import (
    apply_rope_delta,
    rebase_audio_cache,
)


def _frames(start: int, count: int):
    positions = mx.arange(start, start + count, dtype=mx.int32)
    values = positions.astype(mx.float32)
    keys = mx.stack([values, values + 0.25], axis=-1).reshape(1, 1, count, 2)
    return keys, -keys, positions


def test_ring_append_preserves_order_without_reallocating_buffers():
    cache = Audio8LayerCache(capacity=4)
    cache.append(*_frames(0, 2))
    buffer_ids = (id(cache._keys), id(cache._values), id(cache._positions))

    cache.append(*_frames(2, 1))
    cache.append(*_frames(3, 2))

    assert (id(cache._keys), id(cache._values), id(cache._positions)) == buffer_ids
    np.testing.assert_array_equal(np.asarray(cache.positions), [1, 2, 3, 4])
    np.testing.assert_array_equal(
        np.asarray(cache.keys).reshape(-1), [1, 1.25, 2, 2.25, 3, 3.25, 4, 4.25]
    )
    np.testing.assert_array_equal(
        np.asarray(cache.values).reshape(-1),
        [-1, -1.25, -2, -2.25, -3, -3.25, -4, -4.25],
    )


def test_rebase_replaces_all_ring_views_in_the_same_order():
    layer = Audio8LayerCache(capacity=4)
    layer.append(*_frames(0, 6))
    cache = Audio8AudioCache(layers=[layer], position_offset=6)
    old_keys = layer.keys
    old_values = layer.values
    inv_freq = mx.ones((1,), dtype=mx.float32) * 0.01

    rebase_audio_cache(cache, target_position=4, inv_freq=inv_freq)

    expected_keys = apply_rope_delta(old_keys, inv_freq, -2)
    np.testing.assert_allclose(
        np.asarray(layer.keys), np.asarray(expected_keys), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_array_equal(np.asarray(layer.values), np.asarray(old_values))
    np.testing.assert_array_equal(np.asarray(layer.positions), [0, 1, 2, 3])
    assert cache.position_offset == 4
