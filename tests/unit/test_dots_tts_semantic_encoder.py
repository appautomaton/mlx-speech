from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.semantic_encoder import VAESemanticEncoder


def _encoder() -> VAESemanticEncoder:
    mx.random.seed(7)
    return VAESemanticEncoder(
        input_dim=4,
        hidden_size=8,
        output_dim=12,
        num_layers=2,
        num_heads=2,
        intermediate_size=16,
        patch_size=4,
    )


def test_full_prefill_and_incremental_decode_agree() -> None:
    model = _encoder()
    mx.random.seed(11)
    latent = mx.random.normal((1, 8, 4))
    full = model(latent)
    prefill, state = model.prefill(latent[:, :4], max_audio_patches=2)
    decoded, state = model.decode_patch(latent[:, 4:], state)
    combined = mx.concatenate((prefill, decoded), axis=1)
    mx.eval(full, combined)
    assert full.shape == (1, 2, 12)
    assert state.sequence_length == 4
    assert all(cache.offset == 4 for cache in state.layer_caches)
    assert all(cache.capacity == 4 for cache in state.layer_caches)
    # Full masked attention and cached attention use different fused SDPA shapes;
    # gate their equivalent result with the checked-in semantic oracle tolerance.
    np.testing.assert_allclose(combined, full, atol=0.02, rtol=0.02)


def test_semantic_encoder_is_causal_across_patches() -> None:
    model = _encoder()
    mx.random.seed(13)
    latent = mx.random.normal((1, 8, 4))
    changed = mx.concatenate((latent[:, :4], latent[:, 4:] + 10.0), axis=1)
    original_output = model(latent)
    changed_output = model(changed)
    mx.eval(original_output, changed_output)
    np.testing.assert_allclose(
        original_output[:, :1], changed_output[:, :1], atol=2e-5, rtol=2e-5
    )


def test_semantic_inference_fuses_qkv_without_changing_output() -> None:
    model = _encoder()
    latent = mx.random.normal((1, 8, 4))
    expected = model(latent)
    mx.eval(expected)

    model.fuse_for_inference()
    actual = model(latent)
    mx.eval(actual)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    parameters = set(tree_flatten(model.parameters(), destination={}))
    assert "encoder.layers.0.attn.qkv_proj.weight" in parameters
    assert not any("encoder.layers.0.attn.q_proj" in name for name in parameters)
    model.fuse_for_inference()


def test_semantic_state_tracks_conv_tail_and_layer_caches() -> None:
    model = _encoder()
    output, state = model.prefill(
        mx.zeros((2, 4, 4)),
        max_audio_patches=3,
    )
    mx.eval(
        output, state.conv_tail, *(cache.fetch()[0] for cache in state.layer_caches)
    )
    assert state.conv_tail.shape == (2, 1, 4)
    assert len(state.layer_caches) == 2
    assert all(cache.offset == 2 for cache in state.layer_caches)
    assert all(cache.capacity == 6 for cache in state.layer_caches)
    assert all(cache.keys.shape == (2, 6, 2, 4) for cache in state.layer_caches)
    assert all(cache.fetch()[0].shape == (2, 2, 2, 4) for cache in state.layer_caches)


def test_semantic_state_reuses_bounded_storage_and_rejects_overflow() -> None:
    model = _encoder()
    _, state = model.prefill(
        mx.zeros((1, 4, 4)),
        max_audio_patches=2,
    )
    current_state = state
    storage_ids = tuple(
        (id(cache.keys), id(cache.values)) for cache in state.layer_caches
    )
    _, returned_state = model.decode_patch(mx.ones((1, 4, 4)), state)

    assert returned_state is current_state
    assert state is current_state
    assert state.sequence_length == 4
    assert all(cache.offset == state.sequence_length for cache in state.layer_caches)
    assert state.conv_tail.shape == (1, 1, 4)
    assert (
        tuple((id(cache.keys), id(cache.values)) for cache in state.layer_caches)
        == storage_ids
    )
    assert all(cache.capacity == 4 for cache in state.layer_caches)
    with pytest.raises(ValueError, match="overflow"):
        model.decode_patch(mx.ones((1, 4, 4)), state)


def test_semantic_rejected_later_layer_append_leaves_state_unchanged() -> None:
    model = _encoder()
    _, state = model.prefill(
        mx.zeros((1, 4, 4)),
        max_audio_patches=3,
    )
    prior_offsets = tuple(cache.offset for cache in state.layer_caches)
    prior_sequence_length = state.sequence_length
    prior_conv_tail = state.conv_tail

    later_cache = state.layer_caches[-1]
    later_cache.keys = later_cache.keys[:, : later_cache.offset, :, :]
    later_cache.values = later_cache.values[:, : later_cache.offset, :, :]
    later_cache.max_capacity = later_cache.offset
    with pytest.raises(ValueError, match="overflow"):
        model.decode_patch(mx.ones((1, 4, 4)), state)

    assert state.sequence_length == prior_sequence_length
    assert state.conv_tail is prior_conv_tail
    assert tuple(cache.offset for cache in state.layer_caches) == prior_offsets
    assert all(cache.fetch()[0].shape[1] == 2 for cache in state.layer_caches)


def test_semantic_encoder_rejects_misaligned_inputs_and_state() -> None:
    model = _encoder()
    with pytest.raises(ValueError, match="max_audio_patches"):
        model.prefill(mx.zeros((1, 4, 4)), max_audio_patches=0)
    with pytest.raises(ValueError, match="divisible"):
        model(mx.zeros((1, 6, 4)))
    _, state = model.prefill(mx.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="length 4"):
        model.decode_patch(mx.zeros((1, 8, 4)), state)
    with pytest.raises(ValueError, match="batch size"):
        model.decode_patch(mx.zeros((2, 4, 4)), state)
