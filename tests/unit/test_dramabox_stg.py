"""Unit tests for DramaBox STG (Spatio-Temporal Guidance).

Covers the four touch-points that thread the self-attn passthrough from the
attention primitive up to the denoising loop, plus the warm-server default.
No checkpoints required.
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from mlx_speech.models.dramabox.diffusion import LatentState
from mlx_speech.models.dramabox.dit import DiTConfig, LTXModel
from mlx_speech.models.dramabox.ltx.attention import LTXAttention
from mlx_speech.models.dramabox.sampling import GuiderParams, euler_denoising_loop


# --------------------------------------------------------------------------- #
# LTXAttention — skip_self_attn value passthrough
# --------------------------------------------------------------------------- #

def test_attention_skip_equals_value_passthrough_ungated():
    """Ungated skip output is exactly ``to_out(to_v(x))`` (no QK softmax)."""
    mx.random.seed(0)
    attn = LTXAttention(query_dim=8, heads=2, dim_head=4, apply_gated_attention=False)
    x = mx.random.normal((1, 5, 8))

    skip = attn(x, skip_self_attn=True)
    expected = attn.to_out[0](attn.to_v(x))

    assert mx.allclose(skip, expected, atol=1e-5).item()


def test_attention_skip_applies_gate_and_to_out():
    """Gated skip path keeps the per-head gate and to_out around the value."""
    mx.random.seed(1)
    attn = LTXAttention(query_dim=8, heads=2, dim_head=4, apply_gated_attention=True)
    x = mx.random.normal((1, 5, 8))

    skip = attn(x, skip_self_attn=True)

    v = attn.to_v(x).reshape(1, 5, 2, 4)
    gates = 2.0 * mx.sigmoid(attn.to_gate_logits(x))
    gated = (v * gates[..., None]).reshape(1, 5, 8)
    expected = attn.to_out[0](gated)

    assert mx.allclose(skip, expected, atol=1e-5).item()


def test_attention_skip_differs_from_full_attention():
    """The perturbation must actually change the output."""
    mx.random.seed(2)
    attn = LTXAttention(query_dim=8, heads=2, dim_head=4, apply_gated_attention=True)
    x = mx.random.normal((1, 6, 8))

    full = attn(x)  # default skip_self_attn=False
    skip = attn(x, skip_self_attn=True)

    assert not mx.allclose(skip, full, atol=1e-4).item()


def test_attention_skip_ignores_rope():
    """Skip path never touches Q/K, so the RoPE table is irrelevant to it."""
    mx.random.seed(3)
    attn = LTXAttention(query_dim=8, heads=2, dim_head=4, apply_gated_attention=True)
    x = mx.random.normal((1, 5, 8))
    cos = mx.random.normal((1, 5, 2))
    sin = mx.random.normal((1, 5, 2))

    no_rope = attn(x, skip_self_attn=True)
    with_rope = attn(x, skip_self_attn=True, rope_cos_sin=(cos, sin))

    assert mx.allclose(no_rope, with_rope, atol=0.0).item()


# --------------------------------------------------------------------------- #
# LTXModel — stg_blocks routing by block index
# --------------------------------------------------------------------------- #

def _tiny_model() -> tuple[LTXModel, DiTConfig]:
    cfg = DiTConfig(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=3,
        audio_positional_embedding_max_pos=(20,),
    )
    return LTXModel(cfg), cfg


def _tiny_inputs(cfg: DiTConfig):
    B, T, T_text = 1, 4, 3
    x = mx.random.normal((B, T, cfg.audio_in_channels))
    a_ctx = mx.random.normal((B, T_text, cfg.audio_cross_attention_dim))
    sigma = mx.array([0.7], dtype=mx.float32)
    starts = mx.arange(T, dtype=mx.float32).reshape(1, 1, T)
    positions = mx.stack([starts, starts + 1.0], axis=-1)  # (1, 1, T, 2)
    return x, a_ctx, sigma, positions


def test_model_stg_none_equals_empty():
    """``stg_blocks=None`` and ``()`` both run full attention everywhere."""
    mx.random.seed(0)
    model, cfg = _tiny_model()
    x, a_ctx, sigma, positions = _tiny_inputs(cfg)

    out_none = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    out_empty = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, stg_blocks=())

    assert mx.allclose(out_none, out_empty, atol=0.0).item()


def test_model_stg_block_changes_output():
    """Perturbing a real block index changes the velocity."""
    mx.random.seed(0)
    model, cfg = _tiny_model()
    x, a_ctx, sigma, positions = _tiny_inputs(cfg)

    out_none = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    out_b0 = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, stg_blocks=(0,))

    assert not mx.allclose(out_none, out_b0, atol=1e-5).item()


def test_model_stg_out_of_range_block_is_noop():
    """A block index past the stack perturbs nothing."""
    mx.random.seed(0)
    model, cfg = _tiny_model()
    x, a_ctx, sigma, positions = _tiny_inputs(cfg)

    out_none = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions)
    out_oob = model(x, a_ctx=a_ctx, sigma=sigma, positions=positions, stg_blocks=(99,))

    assert mx.allclose(out_none, out_oob, atol=0.0).item()


# --------------------------------------------------------------------------- #
# euler_denoising_loop — ptb pass dispatch
# --------------------------------------------------------------------------- #

class _RecordingX0:
    """Records (a_ctx, stg_blocks) per call; returns the latent unchanged."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, latent, *, a_ctx, sigma, positions=None, rope_cos_sin=None,
                 attention_mask=None, denoise_mask=None, stg_blocks=None):
        self.calls.append((a_ctx, stg_blocks))
        return latent


def _state():
    return LatentState(
        latent=mx.zeros((1, 4, 8), dtype=mx.float32),
        denoise_mask=mx.ones((1, 4, 1), dtype=mx.float32),
        positions=mx.zeros((1, 1, 4, 2), dtype=mx.float32),
        clean_latent=mx.zeros((1, 4, 8), dtype=mx.float32),
        attention_mask=None,
    )


def test_loop_ptb_uses_positive_context_and_stg_blocks():
    """With cfg and stg on: cond, uncond, ptb — ptb carries positive ctx + blocks."""
    pos = mx.zeros((1, 2, 8), dtype=mx.float32)
    neg = mx.ones((1, 2, 8), dtype=mx.float32)
    x0 = _RecordingX0()

    euler_denoising_loop(
        _state(),
        mx.array([1.0, 0.0], dtype=mx.float32),
        x0_model=x0,
        a_ctx=pos,
        a_ctx_neg=neg,
        params=GuiderParams(cfg_scale=2.5, stg_scale=1.5, stg_blocks=(29,),
                            rescale_scale=0.0, modality_scale=1.0),
        positions=_state().positions,
    )

    assert len(x0.calls) == 3
    assert x0.calls[0] == (pos, None)        # cond: positive ctx, no perturbation
    assert x0.calls[1] == (neg, None)        # uncond: negative ctx, no perturbation
    assert x0.calls[2] == (pos, (29,))       # ptb: positive ctx, perturbed block 29


def test_loop_no_ptb_when_stg_zero():
    """stg=0 makes no perturbed pass (baseline path is unchanged)."""
    x0 = _RecordingX0()

    euler_denoising_loop(
        _state(),
        mx.array([1.0, 0.0], dtype=mx.float32),
        x0_model=x0,
        a_ctx=mx.zeros((1, 2, 8), dtype=mx.float32),
        a_ctx_neg=mx.zeros((1, 2, 8), dtype=mx.float32),
        params=GuiderParams(cfg_scale=2.5, stg_scale=0.0,
                            rescale_scale=0.0, modality_scale=1.0),
        positions=_state().positions,
    )

    # cfg on, stg off → cond + uncond only.
    assert len(x0.calls) == 2
    assert all(stg_blocks is None for _, stg_blocks in x0.calls)


# --------------------------------------------------------------------------- #
# generate() default
# --------------------------------------------------------------------------- #

def test_generate_default_stg_matches_warm_server():
    """Default stg_scale is the DramaBox warm-server value (1.5), not the
    old CFG-only fallback (0.0)."""
    from mlx_speech.generation.dramabox import DramaBoxModel

    default = inspect.signature(DramaBoxModel.generate).parameters["stg_scale"].default
    assert default == 1.5
