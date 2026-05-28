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


def test_dit_config_default_audio_inner_dim():
    cfg = DiTConfig()
    assert cfg.audio_inner_dim == 32 * 64  # 2048
    assert cfg.num_layers == 48
