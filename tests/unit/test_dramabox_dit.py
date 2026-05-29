"""Unit tests for the DramaBox DiT primitives.

Stage 6 — timestep embedding, AdaLN, single block shape, full model shape
on a tiny config.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dramabox.dit import DiTConfig, LTXModel
from mlx_speech.models.dramabox.dit.block import LTXBlock
from mlx_speech.models.dramabox.dit.timestep import (
    AdaLayerNormSingle,
    sinusoidal_timestep_embedding,
)


# --------------------------------------------------------------------------- #
# Sinusoidal timestep embedding
# --------------------------------------------------------------------------- #

def test_sinusoidal_timestep_shape():
    t = mx.array([1.0, 5.0, 100.0], dtype=mx.float32)
    emb = sinusoidal_timestep_embedding(t, 256)
    assert emb.shape == (3, 256)


def test_sinusoidal_timestep_distinct_for_distinct_inputs():
    """Different timesteps should produce different embeddings."""
    t = mx.array([1.0, 100.0], dtype=mx.float32)
    emb = sinusoidal_timestep_embedding(t, 64)
    # At least one component differs between the two embeddings
    assert mx.any(mx.abs(emb[0] - emb[1]) > 1e-3).item()


def test_sinusoidal_timestep_per_token_shape_and_equivalence():
    """Per-token ``[B, T]`` input yields ``[B, T, dim]``; a per-token row equals
    the per-batch embedding of the same scalar value."""
    emb_2d = sinusoidal_timestep_embedding(mx.full((2, 3), 5.0, dtype=mx.float32), 64)
    assert emb_2d.shape == (2, 3, 64)
    emb_1d = sinusoidal_timestep_embedding(mx.array([5.0], dtype=mx.float32), 64)
    assert mx.allclose(emb_2d[0, 0], emb_1d[0], atol=0.0).item()


# --------------------------------------------------------------------------- #
# AdaLayerNormSingle
# --------------------------------------------------------------------------- #

def test_adaln_single_output_shape():
    ada = AdaLayerNormSingle(hidden=64, coeff=9)
    t = mx.array([1.0, 5.0], dtype=mx.float32)
    ada_emb, embedded = ada(t, mx.float32)
    assert ada_emb.shape == (2, 9 * 64)
    assert embedded.shape == (2, 64)


def test_adaln_single_prompt_coeff_two():
    ada = AdaLayerNormSingle(hidden=64, coeff=2)
    t = mx.array([1.0], dtype=mx.float32)
    ada_emb, _ = ada(t, mx.float32)
    assert ada_emb.shape == (1, 2 * 64)


def test_adaln_single_per_token_shape():
    """Per-token ``[B, T]`` timesteps preserve the token axis in both outputs."""
    ada = AdaLayerNormSingle(hidden=64, coeff=9)
    t = mx.zeros((2, 5), dtype=mx.float32)
    ada_emb, embedded = ada(t, mx.float32)
    assert ada_emb.shape == (2, 5, 9 * 64)
    assert embedded.shape == (2, 5, 64)


def test_adaln_uniform_per_token_is_uniform_and_matches_scalar():
    """A uniform ``[B, T]`` timestep must produce identical per-token modulation
    for every token (no positional leakage — exact), and each row matches the
    scalar ``[B]`` embedding up to matmul M-shape rounding (loose)."""
    ada = AdaLayerNormSingle(hidden=64, coeff=9)
    scalar_emb, _ = ada(mx.array([500.0], dtype=mx.float32), mx.float32)
    pt_emb, _ = ada(mx.full((1, 4), 500.0, dtype=mx.float32), mx.float32)
    # Every token row identical to row 0 (same matmul → bit-exact).
    for t in range(1, 4):
        assert mx.allclose(pt_emb[0, t], pt_emb[0, 0], atol=0.0).item()
    # Per-token row vs scalar: equal up to cross-shape matmul rounding (a real
    # systematic bug would diverge by order the signal magnitude, not ~1e-3).
    assert mx.allclose(pt_emb[0, 0], scalar_emb[0], atol=2e-3, rtol=2e-3).item()


# --------------------------------------------------------------------------- #
# LTXBlock (single block)
# --------------------------------------------------------------------------- #

def test_ltx_block_output_shape():
    """Build one DiT block and check the output shape matches the input."""
    B, T_audio, T_text, dim = 1, 8, 16, 64
    block = LTXBlock(
        dim=dim, heads=4, dim_head=16, context_dim=dim,
        apply_gated_attention=True, cross_attention_adaln=True,
    )
    x = mx.random.normal((B, T_audio, dim), dtype=mx.float32)
    ada_emb = mx.random.normal((B, 9 * dim), dtype=mx.float32)
    prompt_ada = mx.random.normal((B, 2 * dim), dtype=mx.float32)
    context = mx.random.normal((B, T_text, dim), dtype=mx.float32)
    y = block(
        x,
        ada_emb=ada_emb,
        prompt_ada_emb=prompt_ada,
        context=context,
        rope_cos_sin=None,  # block doesn't apply RoPE if not provided
    )
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# Full LTXModel (tiny config)
# --------------------------------------------------------------------------- #

def test_ltx_model_velocity_shape():
    """Build a tiny LTXModel and check the velocity shape matches the input."""
    cfg = DiTConfig(
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    model = LTXModel(cfg)

    B, T_audio, T_text = 1, 4, 8
    x = mx.random.normal((B, T_audio, 8), dtype=mx.float32)
    a_ctx = mx.random.normal((B, T_text, 8), dtype=mx.float32)
    sigma = mx.array([0.5], dtype=mx.float32)
    positions = mx.broadcast_to(
        mx.arange(T_audio, dtype=mx.float32)[None, None, :, None],
        (B, 1, T_audio, 2),
    )

    velocity = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    assert velocity.shape == (B, T_audio, 8)
    assert mx.all(mx.isfinite(velocity)).item()


def test_ltx_model_all_allow_attention_mask_matches_no_mask():
    cfg = DiTConfig(
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    model = LTXModel(cfg)

    B, T_audio, T_text = 1, 4, 8
    x = mx.random.normal((B, T_audio, 8), dtype=mx.float32)
    a_ctx = mx.random.normal((B, T_text, 8), dtype=mx.float32)
    sigma = mx.array([0.5], dtype=mx.float32)
    positions = mx.broadcast_to(
        mx.arange(T_audio, dtype=mx.float32)[None, None, :, None],
        (B, 1, T_audio, 2),
    )
    all_allow = mx.zeros((B, 1, T_audio, T_audio), dtype=mx.float32)

    no_mask = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    with_mask = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, attention_mask=all_allow)

    assert mx.allclose(with_mask, no_mask, atol=1e-5, rtol=1e-5).item()


def test_ltx_model_blocked_attention_mask_changes_output():
    cfg = DiTConfig(
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    model = LTXModel(cfg)

    B, T_audio, T_text = 1, 4, 8
    x = mx.random.normal((B, T_audio, 8), dtype=mx.float32)
    a_ctx = mx.random.normal((B, T_text, 8), dtype=mx.float32)
    sigma = mx.array([0.5], dtype=mx.float32)
    positions = mx.broadcast_to(
        mx.arange(T_audio, dtype=mx.float32)[None, None, :, None],
        (B, 1, T_audio, 2),
    )
    mask_np = np.zeros((B, 1, T_audio, T_audio), dtype=np.float32)
    mask_np[:, :, 0, 1:] = -1.0e4
    blocked = mx.array(mask_np, dtype=mx.float32)

    no_mask = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    with_mask = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, attention_mask=blocked)
    diff = float(mx.max(mx.abs(with_mask - no_mask)).item())

    assert diff > 1e-6


def _tiny_model_inputs():
    cfg = DiTConfig(
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    B, T_audio, T_text = 1, 4, 8
    x = mx.random.normal((B, T_audio, 8), dtype=mx.float32)
    a_ctx = mx.random.normal((B, T_text, 8), dtype=mx.float32)
    sigma = mx.array([0.5], dtype=mx.float32)
    positions = mx.broadcast_to(
        mx.arange(T_audio, dtype=mx.float32)[None, None, :, None],
        (B, 1, T_audio, 2),
    )
    return LTXModel(cfg), x, a_ctx, sigma, positions, B, T_audio


def test_ltx_model_denoise_mask_none_is_scalar_baseline():
    """The production no-ref path passes denoise_mask=None, which must be the exact
    scalar-sigma code path (bit-identical to omitting the argument)."""
    model, x, a_ctx, sigma, positions, B, T_audio = _tiny_model_inputs()
    omitted = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    explicit_none = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, denoise_mask=None)
    assert mx.allclose(omitted, explicit_none, atol=0.0).item()


def test_ltx_model_frozen_denoise_mask_changes_output():
    """Zeroing a token's denoise_mask (→ timestep 0, clean modulation) measurably
    changes the velocity vs an all-ones mask."""
    model, x, a_ctx, sigma, positions, B, T_audio = _tiny_model_inputs()
    ones = mx.ones((B, T_audio, 1), dtype=mx.float32)
    dm = np.ones((B, T_audio, 1), dtype=np.float32)
    dm[:, -1, :] = 0.0  # freeze last token
    frozen = mx.array(dm)

    base = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, denoise_mask=ones)
    changed = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, denoise_mask=frozen)

    assert float(mx.max(mx.abs(base - changed)).item()) > 1e-6


def test_dit_config_default_audio_inner_dim():
    cfg = DiTConfig()
    assert cfg.audio_inner_dim == 32 * 64  # 2048
    assert cfg.num_layers == 48
