from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

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
    prefill, state = model.prefill(latent[:, :4])
    decoded, state = model.decode_patch(latent[:, 4:], state)
    combined = mx.concatenate((prefill, decoded), axis=1)
    mx.eval(full, combined)
    assert full.shape == (1, 2, 12)
    assert state.sequence_length == 4
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


def test_semantic_state_tracks_conv_tail_and_layer_caches() -> None:
    model = _encoder()
    output, state = model.prefill(mx.zeros((2, 4, 4)))
    mx.eval(output, state.conv_tail, *(cache.keys for cache in state.layer_caches))
    assert state.conv_tail.shape == (2, 1, 4)
    assert len(state.layer_caches) == 2
    assert all(cache.keys.shape == (2, 2, 2, 4) for cache in state.layer_caches)


def test_semantic_encoder_rejects_misaligned_inputs_and_state() -> None:
    model = _encoder()
    with pytest.raises(ValueError, match="divisible"):
        model(mx.zeros((1, 6, 4)))
    _, state = model.prefill(mx.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="length 4"):
        model.decode_patch(mx.zeros((1, 8, 4)), state)
    with pytest.raises(ValueError, match="batch size"):
        model.decode_patch(mx.zeros((2, 4, 4)), state)
