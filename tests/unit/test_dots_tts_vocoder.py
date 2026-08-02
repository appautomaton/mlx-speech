from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.audio_vae import AudioVAE
from mlx_speech.models.dots_tts.vocoder import (
    AMPBlock,
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


@pytest.mark.skipif(
    os.environ.get("MLX_SPEECH_DISABLE_CUSTOM_METAL") == "1"
    or not mx.metal.is_available(),
    reason="requires custom Metal kernels",
)
@pytest.mark.parametrize("length", (1, 7, 33))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16))
def test_fused_alias_free_snakebeta_matches_eager(
    length: int,
    dtype: mx.Dtype,
) -> None:
    mx.random.seed(42 + length)
    activation = AliasFreeSnakeBeta(3)
    activation.set_dtype(dtype)
    activation.alpha = mx.random.normal((3,)) * 0.2
    activation.beta = mx.random.normal((3,)) * 0.2
    value = mx.random.normal((1, length, 3)).astype(dtype)
    expected = activation._eager(value)
    actual = activation(value)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize("kind", ("conv", "transpose", "activation", "amp"))
def test_causal_primitives_preserve_full_sequence_results_across_partitions(
    kind: str,
) -> None:
    mx.random.seed(44)
    if kind == "conv":
        layer = Conv1d(3, 4, 5, dilation=2, causal=True)
    elif kind == "transpose":
        layer = CausalConvTranspose1d(3, 4, 6, stride=3)
    elif kind == "activation":
        layer = AliasFreeSnakeBeta(3)
    else:
        layer = AMPBlock(3, 3, (1, 2))
    value = mx.random.normal((1, 17, 3))
    full = layer(value)
    state = layer.init_stream_state(1, dtype=value.dtype)
    chunks = []
    offset = 0
    for size in (1, 5, 2, 9):
        output, state = layer.stream(value[:, offset : offset + size], state)
        chunks.append(output)
        offset += size
    streamed = mx.concatenate(chunks, axis=1)
    mx.eval(full, streamed, state)

    np.testing.assert_allclose(streamed, full, atol=2e-6, rtol=2e-6)


def test_symmetric_conv_stream_holds_and_flushes_lookahead_exactly_once() -> None:
    mx.random.seed(45)
    convolution = Conv1d(3, 4, 5, causal=False)
    value = mx.random.normal((1, 11, 3))
    full = convolution(value)
    state = convolution.init_lookahead_state(1, dtype=value.dtype)
    outputs = []
    offset = 0
    for size in (1, 4, 2, 4):
        output, state = convolution.stream_lookahead(
            value[:, offset : offset + size],
            state,
            final=False,
        )
        outputs.append(output)
        offset += size
    tail, state = convolution.stream_lookahead(
        value[:, :0],
        state,
        final=True,
    )
    outputs.append(tail)
    streamed = mx.concatenate(outputs, axis=1)
    mx.eval(full, streamed, state)

    assert int(state.pending.shape[1]) == 0
    assert int(tail.shape[1]) == convolution.right_context
    np.testing.assert_allclose(streamed, full, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "partitions",
    ((1,), (1, 2), (1, 1, 1, 8), (3, 5, 3), (11,)),
)
def test_bigvgan_stream_matches_full_for_irregular_and_short_partitions(
    partitions: tuple[int, ...],
) -> None:
    mx.random.seed(46)
    model = AudioVAE(_config(), encoder_residual_layers=1).decoder
    value = mx.random.normal((1, sum(partitions), 4))
    full = model(value)
    state = model.init_stream_state(1)
    outputs = []
    offset = 0
    for index, size in enumerate(partitions):
        output, state = model.stream(
            value[:, offset : offset + size],
            state,
            final=index == len(partitions) - 1,
        )
        outputs.append(output)
        offset += size
    duplicate, duplicate_state = model.stream(value[:, :0], state, final=True)
    streamed = mx.concatenate(outputs, axis=1)
    mx.eval(full, streamed, duplicate, state)

    assert int(duplicate.shape[1]) == 0
    assert duplicate_state is state
    assert state.finalized
    np.testing.assert_allclose(streamed, full, atol=2e-6, rtol=2e-6)


def test_decoder_stream_lookahead_is_derived_from_conv_pre() -> None:
    model = AudioVAE(_config(), encoder_residual_layers=1)
    assert model.decoder.conv_pre.left_context == 2
    assert model.decoder.conv_pre.right_context == 2
    assert model.decoder.resblocks[0][0].left_context == 26
    assert model.decoder.stream_lookahead == 2


def test_bigvgan_rejects_input_outside_its_checkpoint_dtype() -> None:
    model = AudioVAE(_config(), encoder_residual_layers=1)
    model.set_dtype(mx.bfloat16)
    with pytest.raises(ValueError, match="input dtype must match conv_pre weights"):
        model.decoder(mx.zeros((1, 3, model.latent_dim), dtype=mx.float32))


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
