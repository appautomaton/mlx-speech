from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.audio_vae import AudioVAE
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


def test_audio_vae_rejects_invalid_shapes() -> None:
    model = _model()
    with pytest.raises(ValueError, match="waveform shape"):
        model.encode(mx.zeros((1, 2, 16)))
    with pytest.raises(ValueError, match="expects"):
        model.decode(mx.zeros((1, 5, 4)))


def test_official_vocoder_hop_size_is_1920() -> None:
    rates = (2, 2, 2, 4, 6, 10)
    assert int(np.prod(rates)) == 1_920
