from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models.longcat_audiodit.config import LongCatVaeConfig
from mlx_speech.models.longcat_audiodit.vae import LongCatAudioDiTVae


def test_tiny_vae_encode_decode_shapes() -> None:
    config = LongCatVaeConfig(
        latent_dim=4,
        encoder_latent_dim=8,
        channels=4,
        c_mults=(1, 2),
        strides=(2, 2),
        scale=0.5,
        downsampling_ratio=4,
    )
    model = LongCatAudioDiTVae(config)
    audio = mx.zeros((1, 1, 16), dtype=mx.float32)

    latents = model.encode(audio)
    decoded = model.decode(latents)

    assert latents.shape == (1, 4, 4)
    assert decoded.shape == (1, 1, 16)


def test_vae_parameter_tree_matches_checkpoint_naming() -> None:
    config = LongCatVaeConfig(
        latent_dim=4,
        encoder_latent_dim=8,
        channels=4,
        c_mults=(1, 2),
        strides=(2, 2),
        scale=0.5,
        downsampling_ratio=4,
    )
    model = LongCatAudioDiTVae(config)
    params = tree_flatten(model.parameters(), destination={})

    assert "encoder.layers.0.weight" in params
    assert "encoder.layers.1.layers.0.layers.0.alpha" in params
    assert "encoder.layers.1.layers.0.layers.1.weight" in params
    assert "encoder.layers.1.layers.2.layers.3.weight" in params
    assert "decoder.layers.0.weight" in params
    assert "decoder.layers.1.layers.1.weight" in params
    assert "decoder.layers.1.layers.2.layers.1.weight" in params
