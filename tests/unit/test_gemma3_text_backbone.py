"""Unit tests for the Gemma 3 text-only backbone primitives.

These tests run against synthetic small-config models — no checkpoint
required. Checkpoint-aware tests live under `tests/checkpoint/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.gemma3_text import (
    Gemma3Model,
    GemmaTextConfig,
    gemma_rms_norm,
)
from mlx_speech.models.gemma3_text.model import (
    Gemma3MLP,
    GemmaRMSNorm,
    _apply_rope,
    _rope_cos_sin,
)


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

def test_config_from_dict_flat_payload():
    payload = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 100,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
    }
    cfg = GemmaTextConfig.from_dict(payload)
    assert cfg.hidden_size == 64
    assert cfg.num_hidden_layers == 4
    assert cfg.layer_types() == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]


def test_config_from_dict_wrapper_payload():
    payload = {
        "architectures": ["Gemma3Model"],
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 50,
        },
    }
    cfg = GemmaTextConfig.from_dict(payload)
    assert cfg.hidden_size == 32
    assert cfg.head_dim == 8


def test_config_from_dir(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 10,
        },
    }))
    cfg = GemmaTextConfig.from_dir(tmp_path)
    assert cfg.hidden_size == 16


def test_layer_types_match_pattern_six():
    cfg = GemmaTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=12,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=10, sliding_window_pattern=6,
    )
    types = cfg.layer_types()
    full_idxs = [i for i, t in enumerate(types) if t == "full_attention"]
    assert full_idxs == [5, 11]  # 1-indexed positions 6 and 12


def test_layer_types_for_gemma_3_12b():
    """Gemma 3 12B IT: 48 layers, pattern=6 → full attention at 5,11,...,47."""
    cfg = GemmaTextConfig(
        hidden_size=3840, intermediate_size=15360, num_hidden_layers=48,
        num_attention_heads=16, num_key_value_heads=8, head_dim=256,
        vocab_size=262208, sliding_window_pattern=6,
    )
    types = cfg.layer_types()
    full_idxs = [i for i, t in enumerate(types) if t == "full_attention"]
    assert full_idxs == [5, 11, 17, 23, 29, 35, 41, 47]
    assert len(full_idxs) == 8
    assert len([t for t in types if t == "sliding_attention"]) == 40


def test_attention_scaling_uses_query_pre_attn_scalar():
    cfg = GemmaTextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=10, query_pre_attn_scalar=256,
    )
    assert cfg.attention_scaling == 256 ** -0.5


# --------------------------------------------------------------------------- #
# RMSNorm (Gemma 3 variant with `1 + w` offset)
# --------------------------------------------------------------------------- #

def test_gemma_rms_norm_zero_weight_returns_unit_norm():
    """At zero weight, output = (x_normed) * (1 + 0) = x_normed."""
    x = mx.array([[1.0, 2.0, 2.0, 4.0]], dtype=mx.float32)
    w = mx.zeros((4,), dtype=mx.float32)
    y = gemma_rms_norm(x, w, 1e-6)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True))
    expected = x / rms
    assert mx.allclose(y, expected, atol=1e-5).item()


def test_gemma_rms_norm_negative_one_weight_returns_zero():
    """At weight = -1, output = (x_normed) * (1 + (-1)) = 0."""
    x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)
    w = -mx.ones((4,), dtype=mx.float32)
    y = gemma_rms_norm(x, w, 1e-6)
    assert mx.allclose(y, mx.zeros_like(y), atol=1e-6).item()


def test_gemma_rms_norm_preserves_dtype():
    x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.bfloat16)
    w = mx.zeros((4,), dtype=mx.bfloat16)
    y = gemma_rms_norm(x, w, 1e-6)
    assert y.dtype == mx.bfloat16


def test_gemma_rms_norm_module_init_weights_zero():
    norm = GemmaRMSNorm(8, eps=1e-6)
    assert norm.weight.shape == (8,)
    assert mx.all(norm.weight == 0).item()


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #

def test_rope_cos_sin_shape_and_dtype():
    cos, sin = _rope_cos_sin(seq_len=16, head_dim=8, base=10_000.0)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)
    assert cos.dtype == mx.float32


def test_rope_cos_sin_at_position_zero_is_unit():
    """At position 0, cos=1 and sin=0 everywhere."""
    cos, sin = _rope_cos_sin(seq_len=4, head_dim=8, base=10_000.0)
    assert mx.allclose(cos[0], mx.ones((8,), dtype=mx.float32), atol=1e-6).item()
    assert mx.allclose(sin[0], mx.zeros((8,), dtype=mx.float32), atol=1e-6).item()


def test_rope_linear_scaling_divides_inv_freq():
    """Scaling factor 8 should make positions effectively 8x closer in
    frequency space → cos at position 8 with scaling=8 ≈ cos at position 1
    without scaling."""
    cos_unscaled, _ = _rope_cos_sin(seq_len=16, head_dim=8, base=10_000.0, scaling_factor=1.0)
    cos_scaled, _ = _rope_cos_sin(seq_len=16, head_dim=8, base=10_000.0, scaling_factor=8.0)
    # cos at position 8 with scaling=8 ≈ cos at position 1 with no scaling
    assert mx.allclose(cos_scaled[8], cos_unscaled[1], atol=1e-5).item()


def test_apply_rope_preserves_dtype_and_shape():
    B, H, L, D = 1, 2, 4, 8
    q = mx.random.normal((B, H, L, D), dtype=mx.float32)
    k = mx.random.normal((B, H, L, D), dtype=mx.float32)
    cos, sin = _rope_cos_sin(seq_len=L, head_dim=D, base=10_000.0)
    q_out, k_out = _apply_rope(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rope_position_zero_is_identity():
    B, H, D = 1, 1, 8
    q = mx.array([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]], dtype=mx.float32)
    k = mx.array([[[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]], dtype=mx.float32)
    cos, sin = _rope_cos_sin(seq_len=1, head_dim=D, base=10_000.0)
    q_out, k_out = _apply_rope(q, k, cos, sin)
    assert mx.allclose(q_out, q, atol=1e-5).item()
    assert mx.allclose(k_out, k, atol=1e-5).item()


# --------------------------------------------------------------------------- #
# MLP (SwiGLU + tanh-GELU)
# --------------------------------------------------------------------------- #

def test_mlp_output_shape():
    cfg = GemmaTextConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=10,
    )
    mlp = Gemma3MLP(cfg)
    x = mx.random.normal((1, 3, 8), dtype=mx.float32)
    y = mlp(x)
    assert y.shape == (1, 3, 8)


# --------------------------------------------------------------------------- #
# Full forward pass (small synthetic model)
# --------------------------------------------------------------------------- #

def _tiny_config(num_layers: int = 4) -> GemmaTextConfig:
    return GemmaTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        rope_theta=10_000.0,
        rope_local_base_freq=10_000.0,
        sliding_window=8,
        sliding_window_pattern=2,
        query_pre_attn_scalar=16,
    )


def test_model_forward_returns_all_hidden_states():
    cfg = _tiny_config(num_layers=4)
    model = Gemma3Model(cfg)
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    attention_mask = mx.ones((1, 5), dtype=mx.int32)
    out = model(input_ids, attention_mask)

    # Embedding output + 4 layer outputs = 5 tensors
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    for h in out.hidden_states:
        assert h.shape == (1, 5, cfg.hidden_size)
    assert out.last_hidden_state.shape == (1, 5, cfg.hidden_size)


def test_model_forward_with_pad_mask_does_not_nan():
    cfg = _tiny_config(num_layers=2)
    model = Gemma3Model(cfg)
    # Left-pad: first 3 positions are pad, last 2 are real tokens
    input_ids = mx.array([[0, 0, 0, 7, 8]], dtype=mx.int32)
    attention_mask = mx.array([[0, 0, 0, 1, 1]], dtype=mx.int32)
    out = model(input_ids, attention_mask)
    # Real tokens shouldn't produce NaN/Inf
    last = out.last_hidden_state[0, -2:]
    assert mx.all(mx.isfinite(last)).item()


def test_model_forward_no_mask_acts_as_full_causal():
    cfg = _tiny_config(num_layers=2)
    model = Gemma3Model(cfg)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    out_no_mask = model(input_ids, None)
    out_with_mask = model(input_ids, mx.ones((1, 3), dtype=mx.int32))
    assert mx.allclose(out_no_mask.last_hidden_state, out_with_mask.last_hidden_state, atol=1e-5).item()


def test_embed_scale_applied():
    """Embedding output (hidden_states[0]) should be ``embedding * sqrt(H)``."""
    cfg = _tiny_config(num_layers=1)
    model = Gemma3Model(cfg)
    raw_embed = model.embed_tokens(mx.array([[1, 2, 3]], dtype=mx.int32))
    out = model(mx.array([[1, 2, 3]], dtype=mx.int32), None)
    # First hidden state is embeddings * sqrt(H)
    import math
    expected = raw_embed * math.sqrt(cfg.hidden_size)
    assert mx.allclose(out.hidden_states[0], expected, atol=1e-4).item()
