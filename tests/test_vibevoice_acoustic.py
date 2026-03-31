"""Tests for VibeVoice acoustic tokenizer."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.vibevoice.acoustic import (
    Block1D,
    CausalConv1d,
    CausalConvTranspose1d,
    ConvRMSNorm,
    VibeVoiceConvCache,
    VibeVoiceConvDecoder,
    VibeVoiceConvEncoder,
)
from mlx_speech.models.vibevoice.config import VibeVoiceConvTokenizerConfig

MODEL_DIR = Path("models/vibevoice/mlx-int8")
HAS_CHECKPOINT = (MODEL_DIR / "config.json").exists()


class TestCausalConv1d:
    def test_output_shape_no_stride(self):
        conv = CausalConv1d(1, 32, kernel_size=7)
        x = mx.ones((1, 1, 100))
        out = conv(x)
        assert out.shape == (1, 32, 100)

    def test_output_shape_with_stride(self):
        conv = CausalConv1d(32, 64, kernel_size=4, stride=2)
        x = mx.ones((1, 32, 100))
        out = conv(x)
        assert out.shape == (1, 64, 50)

    def test_causal_padding(self):
        """Output should only depend on current and past inputs."""
        conv = CausalConv1d(1, 1, kernel_size=3, bias=False)
        conv.conv.weight = mx.ones((1, 3, 1))
        x = mx.ones((1, 1, 10))
        out = conv(x)
        mx.eval(out)
        # First output sees only 1 input (+ 2 zeros from padding)
        assert float(out[0, 0, 0]) == pytest.approx(1.0, abs=1e-5)
        # Third output sees 3 inputs
        assert float(out[0, 0, 2]) == pytest.approx(3.0, abs=1e-5)

    def test_streaming_matches_non_streaming(self):
        conv = CausalConv1d(4, 8, kernel_size=7)
        x = mx.random.normal((1, 4, 20))

        # Non-streaming
        out_full = conv(x)
        mx.eval(out_full)

        # Streaming: process in two chunks
        cache = VibeVoiceConvCache()
        out1 = conv(x[:, :, :10], cache=cache)
        out2 = conv(x[:, :, 10:], cache=cache)
        out_stream = mx.concatenate([out1, out2], axis=2)
        mx.eval(out_stream)

        diff = mx.abs(out_full - out_stream).max().item()
        assert diff < 1e-4, f"Streaming/non-streaming diff: {diff}"


class TestCausalConvTranspose1d:
    def test_output_shape(self):
        convtr = CausalConvTranspose1d(64, 32, kernel_size=16, stride=8)
        x = mx.ones((1, 64, 5))
        out = convtr(x)
        assert out.shape == (1, 32, 40)  # 5 * 8 = 40

    def test_stride_2(self):
        convtr = CausalConvTranspose1d(32, 16, kernel_size=4, stride=2)
        x = mx.ones((1, 32, 10))
        out = convtr(x)
        assert out.shape == (1, 16, 20)


class TestConvRMSNorm:
    def test_output_shape(self):
        norm = ConvRMSNorm(64)
        x = mx.random.normal((1, 64, 100))
        out = norm(x)
        assert out.shape == x.shape


class TestBlock1D:
    def test_output_shape(self):
        block = Block1D(64, kernel_size=7)
        x = mx.random.normal((1, 64, 50))
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = Block1D(32, kernel_size=7)
        x = mx.zeros((1, 32, 20))
        out = block(x)
        mx.eval(out)
        # With zero input and layer_scale ~1e-6, output should be near zero
        assert mx.abs(out).max().item() < 0.1

    def test_streaming(self):
        block = Block1D(32, kernel_size=7)
        x = mx.random.normal((1, 32, 20))

        out_full = block(x)
        mx.eval(out_full)

        cache = VibeVoiceConvCache()
        out1 = block(x[:, :, :10], cache=cache)
        out2 = block(x[:, :, 10:], cache=cache)
        out_stream = mx.concatenate([out1, out2], axis=2)
        mx.eval(out_stream)

        diff = mx.abs(out_full - out_stream).max().item()
        assert diff < 1e-3, f"Block1D streaming diff: {diff}"


class TestEncoder:
    def test_tiny_config(self):
        cfg = VibeVoiceConvTokenizerConfig(
            vae_dim=8, encoder_ratios=(2, 2), encoder_depths="2-2-2",
            encoder_n_filters=4,
        )
        enc = VibeVoiceConvEncoder(cfg)
        x = mx.random.normal((1, 1, 64))
        out = enc(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # vae_dim


class TestDecoder:
    def test_tiny_config(self):
        cfg = VibeVoiceConvTokenizerConfig(
            vae_dim=8, encoder_ratios=(2, 2), encoder_depths="2-2-2",
            encoder_n_filters=4, decoder_n_filters=4,
        )
        dec = VibeVoiceConvDecoder(cfg)
        x = mx.random.normal((1, 8, 4))  # (B, vae_dim, T_frames)
        out = dec(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1  # channels


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint not available")
class TestRealCheckpoint:
    def test_encoder_output_shape(self):
        from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model

        loaded = load_vibevoice_model(MODEL_DIR, strict=False)
        enc = loaded.model.model.acoustic_tokenizer.encoder
        x = mx.random.normal((1, 1, 24000))  # 1 second at 24kHz
        out = enc(x)
        mx.eval(out)
        # 24000 / 3200 ≈ 7.5 frames
        assert out.shape[0] == 1
        assert out.shape[1] == 64  # vae_dim
        assert out.shape[2] >= 7

    def test_encode_decode_roundtrip_shape(self):
        from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model

        loaded = load_vibevoice_model(MODEL_DIR, strict=False)
        at = loaded.model.model.acoustic_tokenizer
        x = mx.random.normal((1, 1, 24000))
        latent = at.encode(x)
        mx.eval(latent)
        # Decode
        recon = at.decode(latent)
        mx.eval(recon)
        assert recon.shape[0] == 1
        assert recon.shape[1] == 1
        # Reconstructed length should be close to original
        assert abs(recon.shape[2] - 24000) < 3200  # within one frame
