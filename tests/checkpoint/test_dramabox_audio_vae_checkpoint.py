"""Checkpoint loading + forward test for the DramaBox AudioVAE.

Tier-2 test: requires `models/dramabox/dramabox-audio-components.safetensors`.
Skipped automatically if absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.audio_vae import (
    AudioVAE,
    AudioVAEConfig,
    load_audio_vae_weights,
)

AUDIO_COMPONENTS = Path("models/dramabox/dramabox-audio-components.safetensors")

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
