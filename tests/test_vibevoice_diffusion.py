"""Tests for VibeVoice diffusion head and scheduler."""

import mlx.core as mx
import pytest

from mlx_voice.models.vibevoice.diffusion import (
    DPMSolverMultistepScheduler,
    SchedulerOutput,
    TimestepEmbedder,
    VibeVoiceDiffusionHead,
)
from mlx_voice.models.vibevoice.config import VibeVoiceDiffusionConfig


class TestTimestepEmbedder:
    def test_output_shape(self):
        emb = TimestepEmbedder(256)
        t = mx.array([100.0, 500.0])
        out = emb(t)
        assert out.shape == (2, 256)

    def test_different_timesteps_different_output(self):
        emb = TimestepEmbedder(64)
        t1 = mx.array([100.0])
        t2 = mx.array([900.0])
        out1 = emb(t1)
        out2 = emb(t2)
        mx.eval(out1, out2)
        diff = mx.abs(out1 - out2).max().item()
        assert diff > 0.01


class TestDiffusionHead:
    def test_output_shape(self):
        cfg = VibeVoiceDiffusionConfig(hidden_size=64, latent_size=8, head_layers=2)
        head = VibeVoiceDiffusionHead(cfg)
        noisy = mx.random.normal((2, 8))
        t = mx.array([500.0, 500.0])
        cond = mx.random.normal((2, 64))
        out = head(noisy, t, condition=cond)
        assert out.shape == (2, 8)

    def test_no_nan(self):
        cfg = VibeVoiceDiffusionConfig(hidden_size=32, latent_size=4, head_layers=1)
        head = VibeVoiceDiffusionHead(cfg)
        noisy = mx.random.normal((1, 4))
        t = mx.array([999.0])
        cond = mx.random.normal((1, 32))
        out = head(noisy, t, condition=cond)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()


class TestDPMSolverScheduler:
    def test_set_timesteps(self):
        scheduler = DPMSolverMultistepScheduler()
        scheduler.set_timesteps(20)
        assert len(scheduler.timesteps) == 20
        # First timestep should be near max
        assert scheduler.timesteps[0] >= 900
        # Should NOT include t=0 (that causes numerical issues)
        assert 0 not in scheduler.timesteps

    def test_step_output(self):
        scheduler = DPMSolverMultistepScheduler(prediction_type="v_prediction")
        scheduler.set_timesteps(5)
        sample = mx.random.normal((1, 64))
        model_output = mx.random.normal((1, 64))
        out = scheduler.step(model_output, scheduler.timesteps[0], sample)
        assert isinstance(out, SchedulerOutput)
        assert out.prev_sample.shape == (1, 64)

    def test_no_nan_through_all_steps(self):
        scheduler = DPMSolverMultistepScheduler(prediction_type="v_prediction")
        scheduler.set_timesteps(20)
        sample = mx.random.normal((1, 64))

        for t in scheduler.timesteps:
            model_output = mx.random.normal((1, 64)) * 0.1
            out = scheduler.step(model_output, t, sample)
            mx.eval(out.prev_sample)
            assert not mx.any(mx.isnan(out.prev_sample)).item(), f"NaN at t={t}"
            sample = out.prev_sample

    def test_convergence(self):
        """After 20 steps, output should have reasonable magnitude."""
        scheduler = DPMSolverMultistepScheduler(prediction_type="v_prediction")
        scheduler.set_timesteps(20)
        sample = mx.random.normal((1, 64))

        for t in scheduler.timesteps:
            model_output = mx.zeros((1, 64))  # predict zero noise
            out = scheduler.step(model_output, t, sample)
            sample = out.prev_sample

        mx.eval(sample)
        rms = mx.sqrt(mx.mean(sample ** 2)).item()
        assert rms < 100, f"Output RMS too large: {rms}"
