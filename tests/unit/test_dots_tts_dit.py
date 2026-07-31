from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.config import DotsTTSTransformerConfig
from mlx_speech.models.dots_tts.dit import DiT, sinusoidal_embedding


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
        positions=(mx.arange(5, dtype=mx.float32) + 3)[None],
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
