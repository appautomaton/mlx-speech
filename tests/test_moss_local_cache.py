import mlx.core as mx

from mlx_voice.models.moss_local import (
    GlobalLayerKVCache,
    LocalKVCache,
    LocalLayerKVCache,
    MosiTTSModel,
    MossTTSLocalConfig,
    MossTTSLocalModel,
)
from mlx_voice.models.moss_local.model import MossTTSAttention


def _tiny_config() -> MossTTSLocalConfig:
    return MossTTSLocalConfig.from_dict(
        {
            "n_vq": 2,
            "audio_vocab_size": 16,
            "audio_pad_code": 16,
            "local_hidden_size": 12,
            "local_ffn_hidden_size": 32,
            "additional_mlp_ffn_hidden_size": 24,
            "local_num_layers": 2,
            "language_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 128,
                "max_position_embeddings": 64,
            },
        }
    )


def test_global_layer_kv_cache_appends_and_tracks_length() -> None:
    cache = GlobalLayerKVCache(
        batch_size=1,
        num_kv_heads=2,
        max_length=4,
        head_dim=8,
        dtype=mx.bfloat16,
    )
    k1 = mx.ones((1, 2, 2, 8), dtype=mx.bfloat16)
    v1 = mx.zeros((1, 2, 2, 8), dtype=mx.bfloat16)
    k2 = mx.full((1, 2, 1, 8), 2.0, dtype=mx.bfloat16)
    v2 = mx.full((1, 2, 1, 8), 3.0, dtype=mx.bfloat16)

    cache.append(k1, v1)
    assert cache.current_length == 2

    cache.append(k2, v2)
    assert cache.current_length == 3
    keys, values = cache.get()

    assert keys.shape == (1, 2, 3, 8)
    assert values.shape == (1, 2, 3, 8)
    assert keys[:, :, 2:, :].tolist() == k2.tolist()
    assert values[:, :, 2:, :].tolist() == v2.tolist()


def test_local_layer_kv_cache_reset_clears_logical_length() -> None:
    cache = LocalLayerKVCache(
        batch_size=1,
        num_kv_heads=2,
        max_length=4,
        head_dim=8,
        dtype=mx.bfloat16,
    )
    cache.append(
        mx.ones((1, 2, 3, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 3, 8), dtype=mx.bfloat16),
    )
    assert cache.current_length == 3

    cache.reset()
    keys, values = cache.get()
    assert cache.current_length == 0
    assert keys.shape == (1, 2, 0, 8)
    assert values.shape == (1, 2, 0, 8)


def test_cached_attention_prefill_matches_full_sequence() -> None:
    config = _tiny_config().language_config
    attention = MossTTSAttention(config)
    hidden_states = (
        mx.arange(0, 1 * 4 * 32, dtype=mx.float32).reshape(1, 4, 32).astype(mx.bfloat16) / 32.0
    )
    attention_mask = mx.array([[1, 1, 1, 1]], dtype=mx.bool_)
    cache = GlobalLayerKVCache(
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        max_length=4,
        head_dim=config.effective_head_dim,
        dtype=mx.float32,
    )

    full = attention(hidden_states, attention_mask=attention_mask)
    cached = attention.prefill(
        hidden_states,
        layer_cache=cache,
        attention_mask=attention_mask,
    )

    assert cache.current_length == 4
    assert mx.allclose(full.astype(mx.float32), cached.astype(mx.float32), atol=1e-3, rtol=1e-3)


def test_global_decode_step_matches_full_forward_last_token() -> None:
    config = _tiny_config()
    model = MosiTTSModel(config)
    prompt = mx.array(
        [
            [
                [1, 16, 16],
                [2, 16, 16],
                [151652, 16, 16],
            ]
        ],
        dtype=mx.int32,
    )
    next_row = mx.array([[[5, 6, 7]]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    full = model(
        input_ids=mx.concatenate([prompt, next_row], axis=1),
        attention_mask=mx.array([[1, 1, 1, 1]], dtype=mx.bool_),
        n_vq_for_inference=2,
    )
    _, cache = model.prefill(
        input_ids=prompt,
        attention_mask=attention_mask,
        n_vq_for_inference=2,
        max_cache_len=4,
    )
    stepped = model.decode_step(
        input_ids=next_row,
        kv_cache=cache,
        n_vq_for_inference=2,
    )

    assert cache.current_length == 4
    assert mx.allclose(
        stepped.last_hidden_state[:, -1, :].astype(mx.float32),
        full.last_hidden_state[:, -1, :].astype(mx.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_local_decode_step_matches_full_forward_last_position() -> None:
    model = MossTTSLocalModel(_tiny_config())
    local_inputs = (
        mx.arange(0, 1 * 3 * model.config.local_hidden_size, dtype=mx.float32)
        .reshape(1, 3, model.config.local_hidden_size)
        .astype(mx.bfloat16)
        / 16.0
    )
    full = model.forward_local_sequence(local_inputs)
    cache = LocalKVCache.allocate(
        model.local_transformer_config,
        batch_size=1,
        max_length=3,
        dtype=mx.float32,
    )

    for index in range(3):
        stepped = model.decode_local_step(
            local_inputs[:, index : index + 1, :],
            kv_cache=cache,
        )

    assert cache.current_length == 3
    assert mx.allclose(
        stepped.last_hidden_state[:, -1, :].astype(mx.float32),
        full.last_hidden_state[:, -1, :].astype(mx.float32),
        atol=2e-2,
        rtol=2e-2,
    )
