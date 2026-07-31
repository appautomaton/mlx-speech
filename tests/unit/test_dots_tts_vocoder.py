from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_speech.models.dots_tts.audio_vae import AudioVAE
from mlx_speech.models.dots_tts.vocoder import (
    AliasFreeSnakeBeta,
    CausalConvTranspose1d,
    Conv1d,
)
from test_dots_tts_audio_vae import _config


def test_causal_convolution_does_not_see_future_inputs() -> None:
    mx.random.seed(41)
    convolution = Conv1d(2, 3, 3, causal=True)
    value = mx.random.normal((1, 8, 2))
    changed = mx.concatenate((value[:, :4], value[:, 4:] + 100.0), axis=1)
    first = convolution(value)
    second = convolution(changed)
    mx.eval(first, second)
    np.testing.assert_allclose(first[:, :4], second[:, :4], atol=0.0, rtol=0.0)


def test_causal_transposed_convolution_has_exact_stride_length() -> None:
    mx.random.seed(43)
    convolution = CausalConvTranspose1d(4, 2, 6, stride=3)
    output = convolution(mx.ones((1, 5, 4)))
    mx.eval(output)
    assert output.shape == (1, 15, 2)
    assert convolution.left_context == 1


def test_alias_free_snakebeta_preserves_shape_and_is_finite() -> None:
    activation = AliasFreeSnakeBeta(3)
    output = activation(mx.linspace(-2.0, 2.0, 21).reshape(1, 7, 3))
    mx.eval(output)
    assert output.shape == (1, 7, 3)
    assert bool(mx.all(mx.isfinite(output)).item())
    assert activation.left_context == 11


def test_decoder_stream_context_is_derived_from_layer_structure() -> None:
    model = AudioVAE(_config(), encoder_residual_layers=1)
    assert model.decoder.conv_pre.left_context == 2
    assert model.decoder.conv_pre.right_context == 2
    assert model.decoder.resblocks[0][0].left_context == 26
    assert model.decoder.stream_left_context == 28
    assert model.decoder.stream_lookahead == 2
    assert model.decoder.stream_window_size(3) == 33


def test_stateful_chunk_decode_matches_full_waveform() -> None:
    mx.random.seed(47)
    model = AudioVAE(_config(), encoder_residual_layers=1)
    latent = mx.random.normal((1, 4, 6))
    full = model.decode(latent)
    state = model.init_decode_state(maximum_chunk_size=3)
    first, state = model.decode_chunk(latent[:, :, :3], state)
    second, state = model.decode_chunk(latent[:, :, 3:], state, final=True)
    combined = mx.concatenate((first, second), axis=-1)
    mx.eval(full, combined)
    assert state.emitted_frames == state.total_frames == 6
    np.testing.assert_allclose(combined, full, atol=5e-3, rtol=5e-3)
