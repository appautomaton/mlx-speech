from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.audio_vae import (
    AudioVAE,
    SLSTM,
    encoder_logical_workspace_bytes,
)
from mlx_speech.models.dots_tts.config import DotsTTSVocoderConfig


def _config() -> DotsTTSVocoderConfig:
    return DotsTTSVocoderConfig.from_dict(
        {
            "sample_rate": 48_000,
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "upsample_initial_channel": 16,
            "resblock": "1",
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
            "downsample_rates": [2, 2],
            "downsample_channels": [4, 8, 16],
            "activation": "snakebeta",
            "snake_logscale": True,
            "latent_dim": 4,
            "causal": True,
            "mi_num_layers": 1,
            "causal_encoder": True,
            "use_bias_at_final": False,
            "use_tanh_at_final": False,
        }
    )


def _model() -> AudioVAE:
    mx.random.seed(31)
    return AudioVAE(_config(), encoder_residual_layers=2)


def test_audio_vae_encode_decode_shapes_and_waveform_health() -> None:
    model = _model()
    mx.random.seed(37)
    waveform = mx.random.normal((1, 1, 32)) * 0.05
    distribution = model.encode(waveform)
    decoded = model.decode(distribution[:, : model.latent_dim])
    mx.eval(distribution, decoded)
    assert distribution.shape == (1, 8, 8)
    assert decoded.shape == (1, 1, 32)
    assert bool(mx.all(mx.isfinite(decoded)).item())
    assert float(mx.max(mx.abs(decoded)).item()) > 1e-6


def test_audio_vae_decode_is_deterministic() -> None:
    model = _model()
    latent = mx.ones((1, 4, 5))
    first = model.decode(latent)
    second = model.decode(latent)
    mx.eval(first, second)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_tiled_batch_decode_matches_the_full_recurrent_bridge() -> None:
    model = _model()
    model.set_dtype(mx.bfloat16)
    mx.random.seed(38)
    latent = mx.random.normal((1, model.latent_dim, 19)).astype(mx.bfloat16)
    projected = model.post_proj(latent.transpose(0, 2, 1))
    bridged = model.dec_mi_layer(projected).astype(model.decoder.input_dtype)
    decoder_state = model.decoder.init_stream_state(1)
    expected, decoder_state = model.decoder.stream(
        bridged,
        decoder_state,
        final=False,
    )
    tail, _decoder_state = model.decoder.stream(
        bridged[:, :0],
        decoder_state,
        final=True,
    )
    expected = mx.concatenate((expected, tail), axis=1)
    expected = expected.astype(mx.float32).transpose(0, 2, 1)
    actual = model.decode(latent)
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_padded_recurrent_tile_does_not_advance_state_past_valid_length() -> None:
    model = _model()
    model.set_dtype(mx.bfloat16)
    mx.random.seed(38)
    latent = mx.random.normal((1, model.latent_dim, 3)).astype(mx.bfloat16)
    initial = model.init_decode_state(maximum_chunk_size=4).recurrent_state
    tiled, tiled_state = model._execute_recurrent_tiles(
        latent,
        initial,
        use_compiled=True,
    )
    projected = model.post_proj(latent.transpose(0, 2, 1))
    expected, expected_state = model.dec_mi_layer.execute_chunk(projected, initial)
    mx.eval(tiled, tiled_state, expected, expected_state)
    np.testing.assert_allclose(tiled, expected, atol=0.0, rtol=0.0)
    for (actual_h, actual_c), (expected_h, expected_c) in zip(
        tiled_state,
        expected_state,
        strict=True,
    ):
        np.testing.assert_allclose(actual_h, expected_h, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual_c, expected_c, atol=0.0, rtol=0.0)
    recurrent_keys = [
        key for key in model._compiled_vocoder_functions if key.operation == "recurrent"
    ]
    assert len(recurrent_keys) == 1
    assert recurrent_keys[0].shapes[:2] == (
        (1, model.latent_dim, 4),
        (),
    )
    compiled_function = model._compiled_vocoder_functions[recurrent_keys[0]]
    shorter = latent[:, :, :1]
    shorter_state = model.init_decode_state(maximum_chunk_size=4).recurrent_state
    shorter_output, _ = model._execute_recurrent_tiles(
        shorter,
        shorter_state,
        use_compiled=True,
    )
    mx.eval(shorter_output)
    recurrent_cache = {
        key: function
        for key, function in model._compiled_vocoder_functions.items()
        if key.operation == "recurrent"
    }
    assert recurrent_cache == {recurrent_keys[0]: compiled_function}


def test_slstm_chunk_execution_matches_zero_state_full_call() -> None:
    mx.random.seed(39)
    recurrent = SLSTM(5, 2)
    value = mx.random.normal((2, 9, 5))
    full = recurrent(value)
    state = recurrent.initial_state(2, dtype=value.dtype)
    outputs = []
    for start, end in ((0, 2), (2, 6), (6, 9)):
        output, state = recurrent.execute_chunk(value[:, start:end], state)
        outputs.append(output)
    chunked = mx.concatenate(outputs, axis=1)
    mx.eval(full, chunked, state)

    assert len(state) == 2
    assert all(hidden.shape == cell.shape == (2, 5) for hidden, cell in state)
    np.testing.assert_allclose(chunked, full, atol=2e-3, rtol=2e-3)


def test_audio_vae_rejects_invalid_shapes() -> None:
    model = _model()
    with pytest.raises(ValueError, match="waveform shape"):
        model.encode(mx.zeros((1, 2, 16)))
    with pytest.raises(ValueError, match="expects"):
        model.decode(mx.zeros((1, 5, 4)))


def test_official_vocoder_hop_size_is_1920() -> None:
    rates = (2, 2, 2, 4, 6, 10)
    assert int(np.prod(rates)) == 1_920


def test_high_precision_reductions_are_bounded_to_encode() -> None:
    model = _model()
    assert model.audio_encoder.pre_conv.high_precision
    assert all(layer.high_precision for layer in model.audio_encoder.down_convs)
    assert model.audio_encoder.post_conv.high_precision
    assert model.enc_mi_layer.high_precision
    assert model.enc_mi_layer.recurrent.high_precision
    assert model.pre_proj.high_precision
    assert not model.post_proj.high_precision
    assert not model.dec_mi_layer.high_precision
    assert not model.dec_mi_layer.recurrent.high_precision
    assert not model.decoder.conv_pre.high_precision


def test_decoder_uses_bf16_weights_with_an_fp32_slstm_boundary() -> None:
    model = _model()
    model.set_dtype(mx.bfloat16)
    latent = mx.ones((1, model.latent_dim, 4), dtype=mx.float32)
    post_projection = model.post_proj(
        latent.astype(model.post_proj.weight.dtype).transpose(0, 2, 1)
    )
    state = model.init_decode_state(maximum_chunk_size=4)
    decoded, state = model.decode_chunk(latent, state, final=True)
    mx.eval(post_projection, decoded, state)

    assert model.post_proj.weight.dtype == mx.bfloat16
    assert post_projection.dtype == mx.bfloat16
    assert model.dec_mi_layer.recurrent_dtype == mx.float32
    assert all(
        hidden.dtype == cell.dtype == mx.float32
        for hidden, cell in state.recurrent_state
    )
    assert model.decoder.input_dtype == mx.bfloat16
    assert all(
        tensor.dtype == model.decoder.input_dtype
        for tensor in state.decoder_state.arrays()
    )
    assert decoded.dtype == mx.float32


def test_representative_encoder_logical_workspace_is_bounded() -> None:
    payload = _config().to_dict()
    payload.update(
        {
            "downsample_rates": [2, 2, 2, 4, 6, 10],
            "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
            "latent_dim": 128,
            "mi_num_layers": 4,
        }
    )
    config = DotsTTSVocoderConfig.from_dict(payload)
    workspace = encoder_logical_workspace_bytes(
        config, sample_count=round(3.2 * config.sample_rate)
    )
    assert workspace == 25_165_824
    assert workspace < 32 * 1024 * 1024
    with pytest.raises(ValueError, match="dimensions must be positive"):
        encoder_logical_workspace_bytes(config, sample_count=0)
