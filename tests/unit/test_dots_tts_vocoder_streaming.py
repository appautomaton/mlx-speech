from __future__ import annotations

from dataclasses import fields

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.audio_vae import (
    _COMPILED_VOCODER_CACHE_LIMIT,
    AudioVAE,
    VocoderDecodeState,
)
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


def _assert_state_close(
    actual: VocoderDecodeState,
    expected: VocoderDecodeState,
) -> None:
    assert actual.maximum_chunk_size == expected.maximum_chunk_size
    assert actual.total_frames == expected.total_frames
    assert actual.emitted_frames == expected.emitted_frames
    np.testing.assert_allclose(
        actual.decoder_input.astype(mx.float32),
        expected.decoder_input.astype(mx.float32),
        atol=0.0,
        rtol=0.0,
    )
    for actual_layer, expected_layer in zip(
        actual.recurrent_state, expected.recurrent_state, strict=True
    ):
        for actual_tensor, expected_tensor in zip(
            actual_layer, expected_layer, strict=True
        ):
            np.testing.assert_allclose(
                actual_tensor,
                expected_tensor,
                atol=1e-5,
                rtol=1e-5,
            )


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


def test_compiled_common_and_residual_shapes_reuse_pure_tensor_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(127)
    model.set_dtype(mx.bfloat16)
    parameter_names = tuple(name for name, _ in tree_flatten(model.parameters()))
    mx.random.seed(131)
    latent = mx.random.normal((1, model.latent_dim, 28))
    eager_state = model.init_decode_state(maximum_chunk_size=16)
    compiled_state = model.init_decode_state(maximum_chunk_size=16)
    offset = 0
    for chunk_frames in (4, 16, 8):
        chunk = latent[:, :, offset : offset + chunk_frames]
        eager_output, eager_state = model._decode_chunk(
            chunk,
            eager_state,
            final=False,
            use_compiled=False,
        )
        compiled_output, compiled_state = model.decode_chunk(chunk, compiled_state)
        mx.eval(eager_output, compiled_output, eager_state, compiled_state)
        np.testing.assert_allclose(
            compiled_output,
            eager_output,
            atol=1e-5,
            rtol=1e-5,
        )
        _assert_state_close(compiled_state, eager_state)
        offset += chunk_frames

    recurrent_keys = {
        key
        for key in model._compiled_vocoder_functions
        if key.operation == "recurrent"
    }
    decoder_keys = {
        key for key in model._compiled_vocoder_functions if key.operation == "decoder"
    }
    assert {key.shapes[0] for key in recurrent_keys} == {
        (1, model.latent_dim, 4),
        (1, model.latent_dim, 8),
        (1, model.latent_dim, 16),
    }
    assert {key.dtypes[0] for key in recurrent_keys} == {str(latent.dtype)}
    assert {key.dtypes[-4:] for key in recurrent_keys} == {
        (
            str(mx.bfloat16),
            str(mx.bfloat16),
            str(mx.float32),
            str(mx.bfloat16),
        )
    }
    assert {key.model_identity for key in recurrent_keys} == {
        id(model.dec_mi_layer)
    }
    assert len(decoder_keys) == 1
    decoder_key = next(iter(decoder_keys))
    assert decoder_key.shapes == ((1, 46, model.latent_dim),)
    assert decoder_key.dtypes == (str(mx.bfloat16),)
    assert decoder_key.model_identity == id(model.decoder)

    recurrent_cache = {
        key: function
        for key, function in model._compiled_vocoder_functions.items()
        if key.operation == "recurrent"
    }
    empty = mx.zeros((1, model.latent_dim, 0), dtype=latent.dtype)
    eager_tail, eager_state = model._decode_chunk(
        empty,
        eager_state,
        final=True,
        use_compiled=False,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            model,
            "_execute_recurrent_step",
            lambda *args, **kwargs: pytest.fail("flush reran SLSTM recurrence"),
        )
        compiled_tail, compiled_state = model.decode_chunk(
            empty, compiled_state, final=True
        )
    mx.eval(eager_tail, compiled_tail, eager_state, compiled_state)
    np.testing.assert_allclose(compiled_tail, eager_tail, atol=1e-5, rtol=1e-5)
    _assert_state_close(compiled_state, eager_state)
    assert recurrent_cache == {
        key: function
        for key, function in model._compiled_vocoder_functions.items()
        if key.operation == "recurrent"
    }

    cached_functions = dict(model._compiled_vocoder_functions)
    warm_state = model.init_decode_state(maximum_chunk_size=16)
    warm_output, _ = model.decode_chunk(latent[:, :, :4], warm_state)
    mx.eval(warm_output)
    assert model._compiled_vocoder_functions == cached_functions
    assert tuple(name for name, _ in tree_flatten(model.parameters())) == parameter_names


def test_recurrent_tile_failure_leaves_caller_state_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(133)
    model.set_dtype(mx.bfloat16)
    mx.random.seed(135)
    latent = mx.random.normal((1, model.latent_dim, 20))
    state = model.init_decode_state(maximum_chunk_size=20)
    original_step = model._execute_recurrent_step
    calls = 0

    def fail_second_tile(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected recurrent tile failure")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model, "_execute_recurrent_step", fail_second_tile)
    with pytest.raises(RuntimeError, match="injected recurrent tile failure"):
        model.decode_chunk(latent, state, final=True)
    assert calls == 2
    assert state.total_frames == state.emitted_frames == 0
    assert not bool(mx.any(state.decoder_input).item())
    assert all(
        not bool(mx.any(tensor).item())
        for layer_state in state.recurrent_state
        for tensor in layer_state
    )

    monkeypatch.setattr(model, "_execute_recurrent_step", original_step)
    retried, retried_state = model.decode_chunk(latent, state, final=True)
    expected, expected_state = model.decode_chunk(
        latent,
        model.init_decode_state(maximum_chunk_size=20),
        final=True,
    )
    np.testing.assert_allclose(retried, expected, atol=0.0, rtol=0.0)
    _assert_state_close(retried_state, expected_state)


def test_compiled_recurrent_tiles_isolate_interleaved_requests() -> None:
    model = _model(134)
    model.set_dtype(mx.bfloat16)
    mx.random.seed(136)
    latents = {
        "a": mx.random.normal((1, model.latent_dim, 9)),
        "b": mx.random.normal((1, model.latent_dim, 9)),
    }

    expected = {
        request: _stream_chunks(model, latent, (3, 2, 4))[:2]
        for request, latent in latents.items()
    }
    states = {
        request: model.init_decode_state(maximum_chunk_size=4)
        for request in latents
    }
    chunks = {request: [] for request in latents}
    offsets = {request: 0 for request in latents}
    for size in (3, 3, 2, 2, 4, 4):
        request = "a" if len(chunks["a"]) == len(chunks["b"]) else "b"
        start = offsets[request]
        end = start + size
        output, states[request] = model.decode_chunk(
            latents[request][:, :, start:end],
            states[request],
            final=end == 9,
        )
        chunks[request].append(output)
        offsets[request] = end

    for request in latents:
        actual = mx.concatenate(chunks[request], axis=-1)
        expected_output, expected_state = expected[request]
        np.testing.assert_allclose(actual, expected_output, atol=0.0, rtol=0.0)
        _assert_state_close(states[request], expected_state)


def test_compiled_helpers_observe_same_dtype_weight_replacement() -> None:
    model = _model(137)
    model.set_dtype(mx.bfloat16)
    mx.random.seed(139)
    latent = mx.random.normal((1, model.latent_dim, 4))
    recurrent_state = model.init_decode_state(
        maximum_chunk_size=4
    ).recurrent_state
    warm_recurrent, _ = model._execute_recurrent_step(
        latent,
        recurrent_state,
        use_compiled=True,
    )
    decoder_input = mx.random.normal(
        (1, model.decoder.stream_window_size(4), model.latent_dim)
    ).astype(model.decoder.input_dtype)
    warm_decoder = model._execute_decoder_window(
        decoder_input,
        use_compiled=True,
    )
    mx.eval(warm_recurrent, warm_decoder)
    recurrent_key = next(
        key
        for key in model._compiled_vocoder_functions
        if key.operation == "recurrent"
    )
    decoder_key = next(
        key
        for key in model._compiled_vocoder_functions
        if key.operation == "decoder"
    )
    recurrent_function = model._compiled_vocoder_functions[recurrent_key]
    decoder_function = model._compiled_vocoder_functions[decoder_key]

    model.post_proj.weight = mx.zeros_like(model.post_proj.weight)
    updated_recurrent, _ = model._execute_recurrent_step(
        latent,
        recurrent_state,
        use_compiled=True,
    )
    eager_recurrent, _ = model._execute_recurrent_step(
        latent,
        recurrent_state,
        use_compiled=False,
    )
    model.decoder.conv_pre.weight = mx.zeros_like(model.decoder.conv_pre.weight)
    updated_decoder = model._execute_decoder_window(
        decoder_input,
        use_compiled=True,
    )
    eager_decoder = model._execute_decoder_window(
        decoder_input,
        use_compiled=False,
    )
    mx.eval(updated_recurrent, eager_recurrent, updated_decoder, eager_decoder)

    assert model.post_proj.weight.dtype == mx.bfloat16
    assert model.decoder.conv_pre.weight.dtype == mx.bfloat16
    assert model._compiled_vocoder_functions[recurrent_key] is recurrent_function
    assert model._compiled_vocoder_functions[decoder_key] is decoder_function
    assert not np.allclose(
        warm_recurrent.astype(mx.float32),
        updated_recurrent.astype(mx.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not np.allclose(
        warm_decoder.astype(mx.float32),
        updated_decoder.astype(mx.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        updated_recurrent.astype(mx.float32),
        eager_recurrent.astype(mx.float32),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        updated_decoder.astype(mx.float32),
        eager_decoder.astype(mx.float32),
        atol=0.0,
        rtol=0.0,
    )


def test_compiled_cache_is_bounded_and_preserves_common_warm_shapes() -> None:
    model = _model(149)
    model.set_dtype(mx.bfloat16)
    mx.random.seed(151)
    latent = mx.random.normal(
        (1, model.latent_dim, _COMPILED_VOCODER_CACHE_LIMIT + 6)
    )

    for chunk_frames in (4, 16):
        state = model.init_decode_state(maximum_chunk_size=16)
        output, _ = model.decode_chunk(
            latent[:, :, :chunk_frames],
            state,
            final=True,
        )
        mx.eval(output)
    common_cache = {
        key: function
        for key, function in model._compiled_vocoder_functions.items()
        if model._is_common_compile_key(key)
    }
    assert len(common_cache) == 3

    first_varied_keys = set()
    varied_shapes = [
        frames
        for frames in range(1, _COMPILED_VOCODER_CACHE_LIMIT + 5)
        if frames not in (4, 16)
    ]
    for index, chunk_frames in enumerate(varied_shapes):
        state = model.init_decode_state(maximum_chunk_size=chunk_frames)
        output, _ = model.decode_chunk(
            latent[:, :, :chunk_frames],
            state,
            final=True,
        )
        mx.eval(output)
        if index == 0:
            first_varied_keys = set(model._compiled_vocoder_functions) - set(
                common_cache
            )

    assert len(model._compiled_vocoder_functions) == _COMPILED_VOCODER_CACHE_LIMIT
    assert first_varied_keys.isdisjoint(model._compiled_vocoder_functions)
    assert all(
        model._compiled_vocoder_functions.get(key) is function
        for key, function in common_cache.items()
    )

    for chunk_frames in (4, 16):
        state = model.init_decode_state(maximum_chunk_size=16)
        output, _ = model.decode_chunk(
            latent[:, :, :chunk_frames],
            state,
            final=True,
        )
        mx.eval(output)
    assert all(
        model._compiled_vocoder_functions.get(key) is function
        for key, function in common_cache.items()
    )

    model._clear_compiled_vocoder_cache()
    assert not model._compiled_vocoder_functions


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
