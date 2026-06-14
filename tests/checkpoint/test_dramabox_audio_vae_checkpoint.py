"""Checkpoint loading + forward test for the DramaBox AudioVAE.

Tier-2 test: requires `models/dramabox/mlx-bf16/dramabox-audio-components.safetensors`.
Skipped automatically if absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dramabox.audio_vae import (
    AudioVAE,
    AudioVAEConfig,
    load_audio_vae_weights,
)

AUDIO_COMPONENTS = Path("models/dramabox/mlx-bf16/dramabox-audio-components.safetensors")
ENCODE_FIXTURE = Path("tests/fixtures/dramabox/audio_vae_encode_fixture.npz")

pytestmark = pytest.mark.skipif(
    not AUDIO_COMPONENTS.is_file(),
    reason="DramaBox audio-components shard not present",
)


def test_audio_vae_loads_from_audio_components():
    cfg = AudioVAEConfig()  # DramaBox defaults
    vae = AudioVAE(cfg)
    state = mx.load(str(AUDIO_COMPONENTS))
    n = load_audio_vae_weights(vae, state)
    # Expected: 44 encoder + 56 decoder + 2 per_channel_statistics = 102
    assert n == 102

    # Spot-check shapes
    assert vae.encoder.conv_in.conv.weight.shape == (128, 3, 3, 2)  # MLX channel-last: (out, kh, kw, in)
    assert vae.encoder.conv_out.conv.weight.shape == (16, 3, 3, 512)
    assert vae.decoder.conv_in.conv.weight.shape == (512, 3, 3, 8)
    assert vae.decoder.conv_out.conv.weight.shape == (2, 3, 3, 128)
    assert vae.per_channel_statistics.mean_of_means.shape == (128,)
    assert vae.per_channel_statistics.std_of_means.shape == (128,)


def test_audio_vae_decode_forward_does_not_nan():
    """Decode a synthetic latent with the loaded weights — sanity check
    that the output is finite (no NaN/Inf)."""
    cfg = AudioVAEConfig()
    vae = AudioVAE(cfg)
    state = mx.load(str(AUDIO_COMPONENTS))
    load_audio_vae_weights(vae, state)

    # Small latent for speed: [B=1, z_ch=8, T_lat=8, mel_bins_lat=16]
    latent = mx.random.normal((1, 8, 8, 16), dtype=mx.bfloat16)
    out = vae.decode(latent)
    assert out.shape[0] == 1
    assert out.shape[1] == 2  # stereo
    # mel_bins should be cfg.mel_bins = 64 after 2 upsamples (16 → 32 → 64)
    # but the upsample drops first H-row each time so T is irregular; check finiteness only
    assert mx.all(mx.isfinite(out.astype(mx.float32))).item()


def test_audio_vae_encode_matches_upstream_fixture():
    """Encode fixed 10 s mel fixture and compare against upstream torch encoder."""
    cfg = AudioVAEConfig()
    vae = AudioVAE(cfg)
    state = mx.load(str(AUDIO_COMPONENTS))
    load_audio_vae_weights(vae, state)

    fixture = np.load(ENCODE_FIXTURE)
    expected = fixture["latent"].astype(np.float32)
    mel = mx.array(fixture["mel"], dtype=mx.float32)
    latent = vae.encode(mel)
    mx.eval(latent)

    actual = np.asarray(latent.astype(mx.float32))
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()

    actual_by_channel = actual.transpose(1, 0, 2, 3).reshape(actual.shape[1], -1)
    expected_by_channel = expected.transpose(1, 0, 2, 3).reshape(expected.shape[1], -1)
    numerator = np.sum(actual_by_channel * expected_by_channel, axis=1)
    denominator = np.linalg.norm(actual_by_channel, axis=1) * np.linalg.norm(expected_by_channel, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    assert float(np.min(cosine)) >= 0.99
