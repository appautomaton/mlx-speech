from __future__ import annotations

from dataclasses import fields

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.audio_vae import AudioVAE, VocoderDecodeState
from mlx_speech.models.dots_tts.vocoder import AliasFreeSnakeBeta
from test_dots_tts_audio_vae import _config


def _model(seed: int) -> AudioVAE:
    mx.random.seed(seed)
    return AudioVAE(_config(), encoder_residual_layers=1)


def _stream_chunks(
    model: AudioVAE,
    latent: mx.array,
    chunk_sizes: tuple[int, ...],
) -> tuple[mx.array, VocoderDecodeState, tuple[int, ...]]:
    maximum_chunk_size = max(chunk_sizes)
    state = model.init_decode_state(maximum_chunk_size=maximum_chunk_size)
    chunks = []
    emitted_samples = []
    offset = 0
    size_index = 0
    while offset < int(latent.shape[-1]):
        size = min(chunk_sizes[size_index % len(chunk_sizes)], int(latent.shape[-1]) - offset)
        end = offset + size
        output, state = model.decode_chunk(
            latent[:, :, offset:end],
            state,
            final=end == int(latent.shape[-1]),
        )
        chunks.append(output)
        emitted_samples.append(int(output.shape[-1]))
        assert int(state.decoder_input.shape[1]) == model.decoder.stream_window_size(
            maximum_chunk_size
        )
        offset = end
        size_index += 1
    return mx.concatenate(chunks, axis=-1), state, tuple(emitted_samples)


def test_alias_free_left_padding_has_a_finite_safe_seam() -> None:
    mx.random.seed(83)
    activation = AliasFreeSnakeBeta(2)
    value = mx.random.normal((1, 48, 2))
    full = activation(value)
    cut = 17
    cropped = activation(value[:, cut:])
    mx.eval(full, cropped)

    context = activation.left_context
    unsafe_error = np.max(
        np.abs(
            np.asarray(cropped[:, context - 4])
            - np.asarray(full[:, cut + context - 4])
        )
    )
    assert unsafe_error > 1e-5
    np.testing.assert_allclose(
        cropped[:, context - 1 : context + 2],
        full[:, cut + context - 1 : cut + context + 2],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("weight_dtype", "atol", "rtol"),
    (
        (mx.float32, 5e-3, 5e-3),
        (mx.bfloat16, 1e-2, 1e-2),
    ),
)
def test_streaming_window_is_bounded_and_matches_every_chunk_seam(
    weight_dtype: mx.Dtype,
    atol: float,
    rtol: float,
) -> None:
    model = _model(89)
    model.set_dtype(weight_dtype)
    mx.random.seed(97)
    latent = mx.random.normal((1, model.latent_dim, 40))
    full = model.decode(latent)
    streamed, state, chunk_samples = _stream_chunks(model, latent, (1, 3, 2))
    mx.eval(full, streamed, state)

    assert tuple(field.name for field in fields(state)) == (
        "recurrent_state",
        "decoder_input",
        "maximum_chunk_size",
        "total_frames",
        "emitted_frames",
    )
    assert state.total_frames == state.emitted_frames == 40
    assert int(state.decoder_input.shape[1]) == 33
    np.testing.assert_allclose(streamed, full, atol=atol, rtol=rtol)

    seams = np.cumsum([size for size in chunk_samples if size])[:-1]
    for seam in seams:
        start = max(0, int(seam) - 2)
        end = min(int(full.shape[-1]), int(seam) + 2)
        np.testing.assert_allclose(
            streamed[:, :, start:end],
            full[:, :, start:end],
            atol=atol,
            rtol=rtol,
        )


def test_streaming_flushes_lookahead_once_after_partial_groups() -> None:
    model = _model(101)
    mx.random.seed(103)
    latent = mx.random.normal((1, model.latent_dim, 9))
    full = model.decode(latent)
    state = model.init_decode_state(maximum_chunk_size=3)
    chunks = []
    offset = 0
    for size in (1, 3, 2, 3):
        output, state = model.decode_chunk(latent[:, :, offset : offset + size], state)
        chunks.append(output)
        offset += size
    assert offset == int(latent.shape[-1])
    assert state.emitted_frames == 7

    empty = mx.zeros((1, model.latent_dim, 0))
    tail, state = model.decode_chunk(empty, state, final=True)
    idle, state = model.decode_chunk(empty, state)
    duplicate_tail, duplicate_state = model.decode_chunk(empty, state, final=True)
    chunks.append(tail)
    streamed = mx.concatenate(chunks, axis=-1)
    mx.eval(full, streamed, duplicate_tail)

    assert int(tail.shape[-1]) == model.decoder.stream_lookahead * model.hop_size
    assert int(idle.shape[-1]) == 0
    assert int(duplicate_tail.shape[-1]) == 0
    assert duplicate_state.emitted_frames == duplicate_state.total_frames == 9
    np.testing.assert_allclose(streamed, full, atol=5e-3, rtol=5e-3)


def test_streaming_holds_samples_inside_the_decoder_lookahead() -> None:
    model = _model(107)
    mx.random.seed(109)
    latent = mx.random.normal((1, model.latent_dim, 9))
    prefix_frames = 5
    prefix = model.decode(latent[:, :, :prefix_frames])
    full = model.decode(latent)
    mx.eval(prefix, full)

    stable_samples = (
        prefix_frames - model.decoder.stream_lookahead
    ) * model.hop_size
    np.testing.assert_allclose(
        prefix[:, :, stable_samples - 2 : stable_samples],
        full[:, :, stable_samples - 2 : stable_samples],
        atol=0.0,
        rtol=0.0,
    )
    unsafe_frame_error = np.max(
        np.abs(
            np.asarray(prefix[:, :, stable_samples : stable_samples + model.hop_size])
            - np.asarray(full[:, :, stable_samples : stable_samples + model.hop_size])
        )
    )
    assert unsafe_frame_error > 1e-6


def test_streaming_rejects_invalid_capacity_and_oversized_chunks() -> None:
    model = _model(113)
    with pytest.raises(ValueError, match="maximum_chunk_size must be positive"):
        model.init_decode_state(maximum_chunk_size=0)
    state = model.init_decode_state(maximum_chunk_size=2)
    with pytest.raises(ValueError, match="exceeding maximum_chunk_size=2"):
        model.decode_chunk(mx.zeros((1, model.latent_dim, 3)), state)
