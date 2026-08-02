from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.config import DotsTTSTransformerConfig
from mlx_speech.models.dots_tts.dit import (
    AffineFreeLayerNorm,
    DiT,
    RotaryEmbedding,
    TimestepEmbedder,
    sinusoidal_embedding,
)


def _config(*, layers: int = 1) -> DotsTTSTransformerConfig:
    return DotsTTSTransformerConfig.from_dict(
        {
            "num_layers": layers,
            "num_heads": 2,
            "hidden_size": 8,
            "ffn_hidden_size": 16,
            "modulation": True,
            "qkv_bias": False,
            "qk_norm": True,
            "attn_dropout": 0.0,
            "dropout": 0.0,
            "norm_layer": "RMSNorm",
            "alibi_bias": False,
            "rotary_bias": True,
            "rotary_theta": 10_000.0,
        }
    )


def test_sinusoidal_embedding_is_cos_first_and_continuous() -> None:
    embedded = sinusoidal_embedding(mx.array([0.0, 1.0]), 4)
    mx.eval(embedded)
    np.testing.assert_allclose(embedded[0], [1.0, 1.0, 0.0, 0.0], atol=0.0)
    assert not bool(mx.allclose(embedded[0], embedded[1]).item())


def test_timestep_embedding_crosses_the_bfloat16_autocast_boundary() -> None:
    embedder = TimestepEmbedder(8, frequency_embedding_size=8)
    embedder.set_dtype(mx.bfloat16)
    embedded = embedder(mx.array([0.25], dtype=mx.bfloat16))
    mx.eval(embedded)
    assert embedded.dtype == mx.bfloat16


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_fast_affine_free_layer_norm_matches_float32_reference(dtype) -> None:
    mx.random.seed(51)
    value = mx.random.normal((2, 5, 8)).astype(dtype)
    source = value.astype(mx.float32)
    mean = mx.mean(source, axis=-1, keepdims=True)
    variance = mx.mean((source - mean) ** 2, axis=-1, keepdims=True)
    expected = ((source - mean) * mx.rsqrt(variance + 1e-5)).astype(dtype)
    actual = AffineFreeLayerNorm(8)(value)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        atol=5e-7,
        rtol=0.0,
    )


def test_reusable_rotary_geometry_matches_direct_application() -> None:
    mx.random.seed(52)
    rotary = RotaryEmbedding(4, 10_000.0)
    value = mx.random.normal((2, 3, 5, 4))
    frequencies = rotary(mx.arange(5, dtype=mx.float32)[None])
    direct = rotary.apply(value, frequencies)
    cosine, sine = rotary.cos_sin(frequencies)
    reused = rotary.apply_cos_sin(value, cosine, sine)
    mx.eval(direct, reused)
    np.testing.assert_array_equal(reused, direct)


def test_dit_causal_mask_prevents_future_context_leakage() -> None:
    mx.random.seed(53)
    model = DiT(8, 4, _config())
    value = mx.random.normal((1, 6, 8))
    changed = mx.concatenate((value[:, :3], value[:, 3:] + 100.0), axis=1)
    mask = mx.tril(mx.ones((1, 6, 6), dtype=mx.bool_))
    positions = mx.arange(6, dtype=mx.float32)[None]
    timestep = mx.array([0.4])
    first = model(value, timestep, attention_mask=mask, positions=positions)
    second = model(changed, timestep, attention_mask=mask, positions=positions)
    mx.eval(first, second)
    np.testing.assert_allclose(first[:, :3], second[:, :3], atol=2e-5, rtol=2e-5)


def test_dit_uses_positions_and_speaker_conditioning() -> None:
    mx.random.seed(59)
    model = DiT(8, 4, _config())
    value = mx.random.normal((1, 5, 8))
    timestep = mx.array([0.25])
    base = model(
        value,
        timestep,
        positions=mx.arange(5, dtype=mx.float32)[None],
        speaker_condition=mx.zeros((1, 8)),
    )
    changed_positions = model(
        value,
        timestep,
        # RoPE is invariant to a uniform position offset because attention
        # depends on relative positions. Stretch the spacing instead.
        positions=(mx.arange(5, dtype=mx.float32) * 2)[None],
        speaker_condition=mx.zeros((1, 8)),
    )
    changed_speaker = model(
        value,
        timestep,
        positions=mx.arange(5, dtype=mx.float32)[None],
        speaker_condition=mx.ones((1, 8)),
    )
    mx.eval(base, changed_positions, changed_speaker)
    assert float(mx.max(mx.abs(base - changed_positions)).item()) > 1e-7
    assert float(mx.max(mx.abs(base - changed_speaker)).item()) > 1e-7


def test_meanflow_duration_is_required_and_changes_output() -> None:
    mx.random.seed(61)
    model = DiT(8, 4, _config(), meanflow=True)
    value = mx.ones((1, 4, 8))
    timestep = mx.array([0.5])
    with pytest.raises(ValueError, match="duration"):
        model(value, timestep)
    short = model(value, timestep, duration=mx.array([0.25]))
    long = model(value, timestep, duration=mx.array([0.5]))
    mx.eval(short, long)
    assert float(mx.max(mx.abs(short - long)).item()) > 1e-7


def test_dit_rejects_invalid_runtime_shapes() -> None:
    model = DiT(8, 4, _config())
    with pytest.raises(ValueError, match="expects"):
        model(mx.zeros((1, 3, 7)), mx.array([0.0]))
    with pytest.raises(ValueError, match="timesteps"):
        model(mx.zeros((1, 3, 8)), mx.array([0.0, 1.0]))


def test_dit_projected_attention_supports_a_rectangular_cached_prefix() -> None:
    mx.random.seed(71)
    model = DiT(8, 4, _config())
    attention = model.blocks[0].attn
    prefix = mx.random.normal((1, 3, 8))
    tail = mx.random.normal((1, 2, 8))
    prefix_q, prefix_k, prefix_v = attention.project(
        prefix, positions=mx.arange(3, dtype=mx.float32)[None]
    )
    tail_q, tail_k, tail_v = attention.project(
        tail, positions=mx.arange(3, 5, dtype=mx.float32)[None]
    )
    del prefix_q
    mask = mx.ones((1, 2, 5), dtype=mx.bool_)
    cached = attention.attend(
        tail_q,
        mx.concatenate((prefix_k, tail_k), axis=2),
        mx.concatenate((prefix_v, tail_v), axis=2),
        mask=mask,
    )
    full = attention(
        mx.concatenate((prefix, tail), axis=1),
        positions=mx.arange(5, dtype=mx.float32)[None],
    )[:, -2:]
    mx.eval(cached, full)
    np.testing.assert_allclose(cached, full, atol=2e-5, rtol=2e-5)


def test_dit_inference_fuses_qkv_without_changing_projection() -> None:
    mx.random.seed(73)
    model = DiT(8, 4, _config(layers=2))
    value = mx.random.normal((1, 5, 8))
    positions = mx.arange(5, dtype=mx.float32)[None]
    expected = model.blocks[0].attn.project(value, positions=positions)
    expected_output = model(value, mx.array([0.25]), positions=positions)
    mx.eval(expected, expected_output)

    model.fuse_for_inference()
    actual = model.blocks[0].attn.project(value, positions=positions)
    actual_output = model(value, mx.array([0.25]), positions=positions)
    mx.eval(actual, actual_output)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_output, expected_output, atol=1e-6, rtol=1e-6)

    parameters = set(tree_flatten(model.parameters(), destination={}))
    assert "blocks.0.attn.qkv_proj.weight" in parameters
    assert not any("blocks.0.attn.q_proj" in name for name in parameters)
    model.fuse_for_inference()


def test_preparing_dit_modulations_does_not_change_parameter_names() -> None:
    model = DiT(8, 4, _config(layers=2), meanflow=True)
    before = set(tree_flatten(model.parameters(), destination={}))
    condition = model.prepare_condition(
        mx.array([0.25]),
        duration=mx.array([0.5]),
        speaker_condition=mx.ones((1, 8)),
    )
    modulations = model.prepare_modulations(condition)
    after = set(tree_flatten(model.parameters(), destination={}))
    assert len(modulations[0]) == 2
    assert before == after
