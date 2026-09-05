from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fireredtts3.redae import ISTFT, RedAE, RedAEConfig
from mlx_speech.models.qwen3_asr.text_decoder import _make_additive_attention_mask


def _tiny_config() -> RedAEConfig:
    return RedAEConfig(
        audio_patch_size=4,
        audio_sample_rate=24,
        bottleneck_dim=4,
        head_dim=4,
        enc_hidden_size=8,
        enc_intermediate_size=16,
        enc_num_hidden_layers=1,
        enc_max_position_embeddings=64,
        enc_num_attention_heads=2,
        enc_num_key_value_heads=1,
        enc_extra_downsample_rate=2,
        enc_downsample_num_hidden_layers=1,
        dec_hidden_size=8,
        dec_intermediate_size=16,
        dec_num_hidden_layers=1,
        dec_max_position_embeddings=64,
        dec_num_attention_heads=2,
        dec_num_key_value_heads=1,
        vocab_size=32,
        rope_theta=10_000.0,
    )


def test_tiny_redae_encodes_and_decodes_deterministically() -> None:
    mx.random.seed(7)
    model = RedAE(_tiny_config())
    audio = mx.linspace(-0.1, 0.1, 32, dtype=mx.float32)[None]
    first = model.encode(audio)
    second = model.encode(audio)
    decoded = model.decode(first)
    mx.eval(first, second, decoded)
    assert first.shape == (1, 4, 4)
    assert decoded.shape == (1, 32)
    assert bool(mx.all(mx.isfinite(decoded)).item())
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_redae_wires_component_windows_and_keeps_cls_attention_full() -> None:
    model = RedAE(_tiny_config())
    assert model.encoder.qwen3.layers[0].self_attn.sliding_window == 64
    assert model.decoder.qwen3.layers[0].self_attn.sliding_window == 64
    assert model.encoder.downsample.qwen3.layers[0].self_attn.sliding_window is None


def test_redae_padding_uses_25hz_latent_multiple() -> None:
    model = RedAE(_tiny_config())
    padded = model.pad_audio(mx.ones((1, 29)))
    assert padded.shape == (1, 32)
    np.testing.assert_array_equal(np.asarray(padded[:, :3]), 0.0)
    np.testing.assert_array_equal(np.asarray(padded[:, 3:]), 1.0)


def test_redae_sliding_window_masks_future_and_old_keys() -> None:
    mask = _make_additive_attention_mask(
        query_len=6,
        key_len=6,
        query_offset=0,
        dtype=mx.float32,
        use_causal_mask=True,
        sliding_window=3,
    )[0, 0]
    allowed = np.asarray(mask) == 0
    np.testing.assert_array_equal(
        allowed,
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    )


def test_istft_matches_overlap_add_equation() -> None:
    rng = np.random.default_rng(5)
    spectrum = (
        rng.normal(size=(1, 3, 5)) + 1j * rng.normal(size=(1, 3, 5))
    ).astype(np.complex64)
    actual = ISTFT(n_fft=8, hop_length=2)(mx.array(spectrum))
    mx.eval(actual)

    window = np.hanning(9)[:-1].astype(np.float32)
    frames = np.fft.irfft(spectrum, n=8, axis=-1).real * window
    expected = np.zeros((1, 12), dtype=np.float32)
    envelope = np.zeros(12, dtype=np.float32)
    for index in range(3):
        expected[:, index * 2 : index * 2 + 8] += frames[:, index]
        envelope[index * 2 : index * 2 + 8] += window * window
    expected = (expected / np.maximum(envelope, 1e-11))[..., 3:-3]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_redae_rejects_invalid_decoder_shape() -> None:
    model = RedAE(_tiny_config())
    with pytest.raises(ValueError, match="bottleneck_dim"):
        model.decode(mx.zeros((1, 4, 3)))
