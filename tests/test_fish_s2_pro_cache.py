import mlx.core as mx
import pytest
from mlx_speech.models.fish_s2_pro.cache import KVCache


def test_kv_cache_init():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    assert cache.num_layers == 2
    assert cache.dim == 512
    assert cache.max_length == 1024


def test_kv_cache_update():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    key = mx.random.normal((1, 8, 10, 64))  # (batch, heads, seq, dim/heads)
    val = mx.random.normal((1, 8, 10, 64))
    cache.update(0, key, val)
    assert cache.offset == 10


def test_kv_cache_full_update():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    key = mx.random.normal((1, 8, 10, 64))
    val = mx.random.normal((1, 8, 10, 64))
    cache.update(0, key, val)
    keys, vals = cache.get()
    # Should have cached keys and values
    assert keys[0].shape[-2] == 10


def test_kv_cache_get_specific_layer():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    key1 = mx.random.normal((1, 8, 5, 64))
    val1 = mx.random.normal((1, 8, 5, 64))
    cache.update(0, key1, val1)

    key2 = mx.random.normal((1, 8, 7, 64))
    val2 = mx.random.normal((1, 8, 7, 64))
    cache.update(1, key2, val2)

    layer0_keys, layer0_vals = cache.get(layer_idx=0)
    layer1_keys, layer1_vals = cache.get(layer_idx=1)

    assert layer0_keys.shape[-2] == 5
    assert layer1_keys.shape[-2] == 7


def test_kv_cache_reset():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    key = mx.random.normal((1, 8, 10, 64))
    val = mx.random.normal((1, 8, 10, 64))
    cache.update(0, key, val)
    assert cache.offset == 10

    cache.reset()
    assert cache.offset == 0


def test_kv_cache_trim():
    cache = KVCache(num_layers=2, dim=512, max_length=1024)
    key = mx.random.normal((1, 8, 20, 64))
    val = mx.random.normal((1, 8, 20, 64))
    cache.update(0, key, val)
    assert cache.offset == 20

    cache.trim_to(15)
    assert cache.offset == 15
