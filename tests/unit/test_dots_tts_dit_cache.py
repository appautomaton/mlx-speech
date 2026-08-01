from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.dots_tts.config import DotsTTSTransformerConfig
from mlx_speech.models.dots_tts.dit import DiT
from mlx_speech.models.dots_tts.dit_inference import (
    CachedMeanFlowSolver,
    CachedSOARSolver,
    DiTKvCache,
    resolve_dit_cache_bucket,
)
from mlx_speech.models.dots_tts.solvers import MeanFlowSolver, SOARSolver


_HIDDEN_SIZE = 8
_LATENT_DIM = 3
_PATCH_SIZE = 2
_UNIT_LENGTH = 3


def _config(*, layers: int = 2) -> DotsTTSTransformerConfig:
    return DotsTTSTransformerConfig.from_dict(
        {
            "num_layers": layers,
            "num_heads": 2,
            "hidden_size": _HIDDEN_SIZE,
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


def _model_and_projection(*, meanflow: bool) -> tuple[DiT, nn.Linear]:
    model = DiT(
        _HIDDEN_SIZE,
        _LATENT_DIM,
        _config(),
        meanflow=meanflow,
        frequency_embedding_size=8,
    )
    projection = nn.Linear(_LATENT_DIM, _HIDDEN_SIZE, bias=True)
    return model, projection


def _unit(start: float) -> mx.array:
    values = mx.arange(_UNIT_LENGTH * _HIDDEN_SIZE, dtype=mx.float32)
    return (values.reshape(1, _UNIT_LENGTH, _HIDDEN_SIZE) / 50.0) + start


def _current(start: float) -> mx.array:
    return mx.full((1, 1, _HIDDEN_SIZE), start, dtype=mx.float32)


def _sequence(history: list[mx.array], current: mx.array) -> mx.array:
    padding = mx.zeros((1, _PATCH_SIZE, _HIDDEN_SIZE), dtype=current.dtype)
    return mx.concatenate((*history, current, padding), axis=1)


def _mask(sequence_length: int) -> mx.array:
    fm_length = sequence_length - _PATCH_SIZE
    block_start = fm_length - 1
    rows = []
    if block_start:
        causal = mx.arange(block_start)[:, None] >= mx.arange(block_start)[None]
        rows.append(
            mx.concatenate(
                (
                    causal,
                    mx.zeros(
                        (block_start, sequence_length - block_start),
                        dtype=mx.bool_,
                    ),
                ),
                axis=1,
            )
        )
    rows.append(
        mx.ones((sequence_length - block_start, sequence_length), dtype=mx.bool_)
    )
    return mx.concatenate(rows, axis=0)[None]


def _positions(sequence_length: int) -> mx.array:
    return mx.arange(sequence_length, dtype=mx.float32)[None]


def _meanflow_pair() -> tuple[MeanFlowSolver, CachedMeanFlowSolver]:
    model, projection = _model_and_projection(meanflow=True)
    return (
        MeanFlowSolver(model, projection, latent_dim=_LATENT_DIM),
        CachedMeanFlowSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        ),
    )


def _soar_pair() -> tuple[SOARSolver, CachedSOARSolver]:
    model, projection = _model_and_projection(meanflow=False)
    return (
        SOARSolver(model, projection, latent_dim=_LATENT_DIM),
        CachedSOARSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        ),
    )


def _published_cache(
    solver: CachedMeanFlowSolver | CachedSOARSolver,
    *,
    capacity_patches: int,
    published_patches: int,
) -> DiTKvCache:
    cache = solver.runner.allocate_cache(
        capacity_patches=capacity_patches,
        nfe=2,
        batch_size=1,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )
    published_tokens = published_patches * solver.unit_length
    for nfe_index in range(cache.nfe):
        shape = cache.cache_k[nfe_index, ..., :published_tokens, :].shape
        size = int(np.prod(shape))
        values = mx.arange(size, dtype=mx.float32).reshape(shape)
        values = values / 200.0 + nfe_index / 10.0
        cache.cache_k[nfe_index, ..., :published_tokens, :] = values
        cache.cache_v[nfe_index, ..., :published_tokens, :] = -values
    cache.offsets = [published_tokens] * cache.nfe
    mx.eval(cache.cache_k, cache.cache_v)
    return cache


def _copy_cache_to_capacity(
    solver: CachedMeanFlowSolver | CachedSOARSolver,
    source: DiTKvCache,
    capacity_patches: int,
) -> DiTKvCache:
    cache = solver.runner.allocate_cache(
        capacity_patches=capacity_patches,
        nfe=source.nfe,
        batch_size=source.batch_size,
        key_dtype=source.cache_k.dtype,
        value_dtype=source.cache_v.dtype,
    )
    for nfe_index, offset in enumerate(source.offsets):
        cache.cache_k[nfe_index, ..., :offset, :] = source.cache_k[
            nfe_index, ..., :offset, :
        ]
        cache.cache_v[nfe_index, ..., :offset, :] = source.cache_v[
            nfe_index, ..., :offset, :
        ]
    cache.offsets = list(source.offsets)
    mx.eval(cache.cache_k, cache.cache_v)
    return cache


def _oracle_meanflow(
    solver: MeanFlowSolver,
    sequence: mx.array,
    noise: mx.array,
    speaker: mx.array,
) -> mx.array:
    return solver.sample(
        sequence=sequence,
        attention_mask=_mask(int(sequence.shape[1])),
        positions=_positions(int(sequence.shape[1])),
        speaker_condition=speaker,
        steps=2,
        patch_size=_PATCH_SIZE,
        noise=noise,
    )


def _oracle_soar(
    solver: SOARSolver,
    sequence: mx.array,
    cfg_sequence: mx.array,
    noise: mx.array,
    speaker: mx.array,
) -> mx.array:
    return solver.sample(
        sequence=sequence,
        cfg_sequence=cfg_sequence,
        attention_mask=_mask(int(sequence.shape[1])),
        positions=_positions(int(sequence.shape[1])),
        speaker_condition=speaker,
        guidance_scale=1.3,
        steps=2,
        patch_size=_PATCH_SIZE,
        noise=noise,
    )


def test_meanflow_cache_matches_first_and_multiple_full_history_patches() -> None:
    mx.random.seed(73)
    oracle, cached = _meanflow_pair()
    state = cached.new_state(12)
    speaker = mx.arange(_HIDDEN_SIZE, dtype=mx.float32)[None] / 20.0
    history: list[mx.array] = []
    for index in range(3):
        sequence = _sequence(history, _current(0.2 + index))
        noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.1 * (index + 1))
        expected = _oracle_meanflow(oracle, sequence, noise, speaker)
        actual = cached.sample(
            state,
            sequence=sequence,
            speaker_condition=speaker,
            steps=2,
            noise=noise,
        )
        mx.eval(expected, actual)
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
        history.append(_unit(0.4 + index))
    assert state.cache is not None
    assert state.cache.offsets == [2 * _UNIT_LENGTH] * 2


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_compact_tail_matches_full_history_after_cache_creation(mode: str) -> None:
    mx.random.seed(75)
    if mode == "meanflow":
        _oracle, cached = _meanflow_pair()
    else:
        _oracle, cached = _soar_pair()
    full_state = cached.new_state(64)
    compact_state = cached.new_state(64)
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2)
    first_previous = _unit(0.2)
    second_previous = _unit(0.6)
    first_cfg = _unit(-0.2)
    second_cfg = _unit(-0.6)
    second_sequence = _sequence([first_previous], _current(0.4))
    second_cfg_sequence = _sequence([first_cfg], _current(-0.4))
    second_noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.15)
    seed_kwargs = {
        "sequence": second_sequence,
        "speaker_condition": speaker,
        "steps": 2,
        "noise": second_noise,
    }
    if mode == "soar":
        seed_kwargs.update(
            cfg_sequence=second_cfg_sequence,
            guidance_scale=1.3,
        )
    cached.sample(full_state, **seed_kwargs)
    cached.sample(compact_state, **seed_kwargs)

    current = _current(0.9)
    cfg_current = _current(-0.9)
    noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.25)
    full_kwargs = {
        "sequence": _sequence([first_previous, second_previous], current),
        "speaker_condition": speaker,
        "steps": 2,
        "noise": noise,
    }
    tail_kwargs = {
        "previous_unit": second_previous,
        "current_hidden": current,
        "speaker_condition": speaker,
        "steps": 2,
        "noise": noise,
    }
    if mode == "soar":
        full_kwargs.update(
            cfg_sequence=_sequence([first_cfg, second_cfg], cfg_current),
            guidance_scale=1.3,
        )
        tail_kwargs.update(
            cfg_previous_unit=second_cfg,
            cfg_current_hidden=cfg_current,
            guidance_scale=1.3,
        )
    expected = cached.sample(full_state, **full_kwargs)
    actual = cached.sample_tail(compact_state, **tail_kwargs)
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert full_state.cache is not None and compact_state.cache is not None
    assert compact_state.cache.offsets == full_state.cache.offsets


def test_continuation_prefill_is_per_nfe_and_commits_the_final_prompt_unit() -> None:
    mx.random.seed(79)
    oracle, cached = _meanflow_pair()
    state = cached.new_state(64)
    speaker = mx.ones((1, _HIDDEN_SIZE), dtype=mx.float32) * 0.15
    history = [_unit(0.1), _unit(0.5), _unit(0.9)]
    sequence = _sequence(history, _current(1.3))
    noise = mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 0.2
    expected = _oracle_meanflow(oracle, sequence, noise, speaker)
    actual = cached.sample(
        state,
        sequence=sequence,
        speaker_condition=speaker,
        steps=2,
        noise=noise,
    )
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    cache = state.cache
    assert cache is not None
    assert cache.offsets == [3 * _UNIT_LENGTH, 3 * _UNIT_LENGTH]
    assert not bool(
        mx.allclose(
            cache.cache_k[0, :, :, :, : cache.valid_tokens],
            cache.cache_k[1, :, :, :, : cache.valid_tokens],
        ).item()
    )
    np.testing.assert_array_equal(
        cache.cache_k[..., cache.valid_tokens :, :],
        mx.zeros_like(cache.cache_k[..., cache.valid_tokens :, :]),
    )


def test_current_noisy_tail_never_changes_committed_cache_contents() -> None:
    mx.random.seed(83)
    _oracle, cached = _meanflow_pair()
    speaker = mx.ones((1, _HIDDEN_SIZE), dtype=mx.float32) * 0.2
    sequence = _sequence([_unit(0.3)], _current(0.7))
    caller_mask = mx.ones((1, sequence.shape[1], sequence.shape[1]), dtype=mx.bool_)
    first_state = cached.new_state(64)
    second_state = cached.new_state(64)
    cached.sample(
        first_state,
        sequence=sequence,
        attention_mask=caller_mask,
        speaker_condition=speaker,
        steps=2,
        noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    )
    cached.sample(
        second_state,
        sequence=sequence,
        attention_mask=caller_mask,
        speaker_condition=speaker,
        steps=2,
        noise=mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 5.0,
    )
    assert first_state.cache is not None and second_state.cache is not None
    assert first_state.cache.offsets == second_state.cache.offsets == [3, 3]
    np.testing.assert_allclose(
        first_state.cache.cache_k[..., :3, :],
        second_state.cache.cache_k[..., :3, :],
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        first_state.cache.cache_v[..., :3, :],
        second_state.cache.cache_v[..., :3, :],
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_array_equal(
        first_state.cache.cache_k[..., 3:6, :],
        mx.zeros_like(first_state.cache.cache_k[..., 3:6, :]),
    )


def test_soar_cache_matches_cfg_oracle_and_separates_branches() -> None:
    mx.random.seed(89)
    oracle, cached = _soar_pair()
    state = cached.new_state(64)
    speaker = mx.ones((1, _HIDDEN_SIZE), dtype=mx.float32) * 0.25
    first_sequence = _sequence([], _current(0.1))
    first_cfg_sequence = _sequence([], _current(-0.1))
    first_noise = mx.zeros((1, _PATCH_SIZE, _LATENT_DIM))
    first_expected = _oracle_soar(
        oracle, first_sequence, first_cfg_sequence, first_noise, speaker
    )
    first_actual = cached.sample(
        state,
        sequence=first_sequence,
        cfg_sequence=first_cfg_sequence,
        speaker_condition=speaker,
        guidance_scale=1.3,
        steps=2,
        noise=first_noise,
    )
    mx.eval(first_expected, first_actual)
    np.testing.assert_allclose(first_actual, first_expected, atol=3e-5, rtol=3e-5)
    assert state.cache is None

    sequence = _sequence([_unit(0.4)], _current(0.8))
    cfg_sequence = _sequence([_unit(-0.6)], _current(-0.2))
    noise = mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 0.3
    expected = _oracle_soar(oracle, sequence, cfg_sequence, noise, speaker)
    actual = cached.sample(
        state,
        sequence=sequence,
        cfg_sequence=cfg_sequence,
        speaker_condition=speaker,
        guidance_scale=1.3,
        steps=2,
        noise=noise,
    )
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    cache = state.cache
    assert cache is not None
    assert cache.cache_k.shape[:3] == (2, 2, 2)
    assert cache.offsets == [3, 3]
    assert not bool(
        mx.allclose(
            cache.cache_k[0, :, 0, :, :3],
            cache.cache_k[0, :, 1, :, :3],
        ).item()
    )


def test_batched_soar_expands_masks_and_positions_to_cfg_branches() -> None:
    mx.random.seed(91)
    oracle, cached = _soar_pair()
    first = _sequence([_unit(0.2)], _current(0.6))
    second = _sequence([_unit(0.8)], _current(1.2))
    sequence = mx.concatenate((first, second), axis=0)
    first_cfg = _sequence([_unit(-0.2)], _current(-0.6))
    second_cfg = _sequence([_unit(-0.8)], _current(-1.2))
    cfg_sequence = mx.concatenate((first_cfg, second_cfg), axis=0)
    total_length = int(sequence.shape[1])
    attention_mask = mx.concatenate((_mask(total_length), _mask(total_length)), axis=0)
    base_positions = _positions(total_length)
    positions = mx.concatenate((base_positions, base_positions + 2.0), axis=0)
    speaker = mx.concatenate(
        (
            mx.full((1, _HIDDEN_SIZE), 0.1),
            mx.full((1, _HIDDEN_SIZE), 0.3),
        ),
        axis=0,
    )
    noise = (
        mx.arange(2 * _PATCH_SIZE * _LATENT_DIM, dtype=mx.float32).reshape(
            2, _PATCH_SIZE, _LATENT_DIM
        )
        / 20.0
    )
    expected = oracle.sample(
        sequence=sequence,
        cfg_sequence=cfg_sequence,
        attention_mask=attention_mask,
        positions=positions,
        speaker_condition=speaker,
        guidance_scale=1.3,
        steps=2,
        patch_size=_PATCH_SIZE,
        noise=noise,
    )
    state = cached.new_state(64)
    actual = cached.sample(
        state,
        sequence=sequence,
        cfg_sequence=cfg_sequence,
        attention_mask=attention_mask,
        positions=positions,
        speaker_condition=speaker,
        guidance_scale=1.3,
        steps=2,
        noise=noise,
    )
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert state.cache is not None
    assert state.cache.cache_k.shape[2] == 4
    assert state.cache.offsets == [_UNIT_LENGTH, _UNIT_LENGTH]


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_bfloat16_later_patch_uses_projected_cache_dtypes(mode: str) -> None:
    mx.random.seed(93)
    model, projection = _model_and_projection(meanflow=mode == "meanflow")
    model.set_dtype(mx.bfloat16)
    projection.set_dtype(mx.bfloat16)
    if mode == "meanflow":
        oracle = MeanFlowSolver(model, projection, latent_dim=_LATENT_DIM)
        cached = CachedMeanFlowSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        )
    else:
        oracle = SOARSolver(model, projection, latent_dim=_LATENT_DIM)
        cached = CachedSOARSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        )
    state = cached.new_state(64)
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2, dtype=mx.bfloat16)
    first_sequence = _sequence([], _current(0.1).astype(mx.bfloat16))
    first_cfg = _sequence([], _current(-0.1).astype(mx.bfloat16))
    first_noise = mx.zeros((1, _PATCH_SIZE, _LATENT_DIM), dtype=mx.bfloat16)
    if mode == "meanflow":
        first_expected = _oracle_meanflow(oracle, first_sequence, first_noise, speaker)
        first_actual = cached.sample(
            state,
            sequence=first_sequence,
            speaker_condition=speaker,
            steps=2,
            noise=first_noise,
        )
    else:
        first_expected = _oracle_soar(
            oracle, first_sequence, first_cfg, first_noise, speaker
        )
        first_actual = cached.sample(
            state,
            sequence=first_sequence,
            cfg_sequence=first_cfg,
            speaker_condition=speaker,
            guidance_scale=1.3,
            steps=2,
            noise=first_noise,
        )
    mx.eval(first_expected, first_actual)
    np.testing.assert_allclose(
        first_actual.astype(mx.float32),
        first_expected.astype(mx.float32),
        atol=5e-2,
        rtol=5e-2,
    )

    previous = _unit(0.4).astype(mx.bfloat16)
    cfg_previous = _unit(-0.4).astype(mx.bfloat16)
    sequence = _sequence([previous], _current(0.8).astype(mx.bfloat16))
    cfg_sequence = _sequence([cfg_previous], _current(-0.2).astype(mx.bfloat16))
    noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.3, dtype=mx.bfloat16)
    if mode == "meanflow":
        expected = _oracle_meanflow(oracle, sequence, noise, speaker)
        actual = cached.sample(
            state,
            sequence=sequence,
            speaker_condition=speaker,
            steps=2,
            noise=noise,
        )
        cfg_previous_unit = None
        cfg_current_hidden = None
    else:
        expected = _oracle_soar(oracle, sequence, cfg_sequence, noise, speaker)
        actual = cached.sample(
            state,
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            speaker_condition=speaker,
            guidance_scale=1.3,
            steps=2,
            noise=noise,
        )
        cfg_previous_unit = cfg_previous
        cfg_current_hidden = cfg_sequence[:, _UNIT_LENGTH : _UNIT_LENGTH + 1]
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    cache = state.cache
    assert cache is not None
    assert cache.offsets == [_UNIT_LENGTH, _UNIT_LENGTH]
    assert state.modulations_by_nfe is not None
    _velocity, projected_keys, projected_values = cached.runner.next_velocity(
        noise,
        previous_unit=previous,
        current_hidden=sequence[:, _UNIT_LENGTH : _UNIT_LENGTH + 1],
        cfg_previous_unit=cfg_previous_unit,
        cfg_current_hidden=cfg_current_hidden,
        cache=None,
        nfe_index=0,
        modulations=state.modulations_by_nfe[0],
        positions=_positions(2 * _UNIT_LENGTH),
        attention_mask=_mask(2 * _UNIT_LENGTH),
        guidance_scale=1.3,
    )
    assert cache.cache_k.dtype == projected_keys.dtype
    assert cache.cache_v.dtype == projected_values.dtype
    assert bool(mx.any(cache.cache_k[..., :_UNIT_LENGTH, :] != 0).item())

    next_previous = _unit(0.9).astype(mx.bfloat16)
    next_cfg_previous = _unit(-0.9).astype(mx.bfloat16)
    later_sequence = _sequence(
        [previous, next_previous], _current(1.2).astype(mx.bfloat16)
    )
    later_cfg_sequence = _sequence(
        [cfg_previous, next_cfg_previous],
        _current(-0.6).astype(mx.bfloat16),
    )
    later_noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.45, dtype=mx.bfloat16)
    if mode == "meanflow":
        later_expected = _oracle_meanflow(oracle, later_sequence, later_noise, speaker)
        later_actual = cached.sample(
            state,
            sequence=later_sequence,
            speaker_condition=speaker,
            steps=2,
            noise=later_noise,
        )
    else:
        later_expected = _oracle_soar(
            oracle,
            later_sequence,
            later_cfg_sequence,
            later_noise,
            speaker,
        )
        later_actual = cached.sample(
            state,
            sequence=later_sequence,
            cfg_sequence=later_cfg_sequence,
            speaker_condition=speaker,
            guidance_scale=1.3,
            steps=2,
            noise=later_noise,
        )
    mx.eval(later_expected, later_actual)
    np.testing.assert_allclose(
        later_actual.astype(mx.float32),
        later_expected.astype(mx.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    assert cache.offsets == [2 * _UNIT_LENGTH, 2 * _UNIT_LENGTH]
    assert cache.cache_k.dtype == projected_keys.dtype
    assert cache.cache_v.dtype == projected_values.dtype


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1, 64), (64, 64), (65, 128), (129, 256), (257, 512), (512, 512)],
)
def test_cache_bucket_resolution(requested: int, expected: int) -> None:
    assert resolve_dit_cache_bucket(requested) == expected


def test_default_request_allocates_physical_bucket_on_second_patch() -> None:
    _oracle, cached = _meanflow_pair()
    state = cached.new_state(500)
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2)
    noise = mx.zeros((1, _PATCH_SIZE, _LATENT_DIM))

    cached.sample(
        state,
        sequence=_sequence([], _current(0.2)),
        speaker_condition=speaker,
        steps=2,
        noise=noise,
    )
    assert state.max_patches == 500
    assert state.capacity_patches == 512
    assert state.cache is None

    cached.sample(
        state,
        sequence=_sequence([_unit(0.4)], _current(0.6)),
        speaker_condition=speaker,
        steps=2,
        noise=noise,
    )
    assert state.cache is not None
    assert state.cache.capacity_patches == 64
    assert state.cache.offsets == [_UNIT_LENGTH, _UNIT_LENGTH]


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
@pytest.mark.parametrize(
    ("source_capacity", "target_capacity"),
    ((64, 128), (128, 256), (256, 512)),
)
def test_cache_grows_transactionally_without_changing_solver_output(
    mode: str,
    source_capacity: int,
    target_capacity: int,
) -> None:
    _oracle, cached = _meanflow_pair() if mode == "meanflow" else _soar_pair()
    source = _published_cache(
        cached,
        capacity_patches=source_capacity,
        published_patches=source_capacity,
    )
    baseline = _copy_cache_to_capacity(cached, source, target_capacity)
    source_keys = source.cache_k + 0
    source_values = source.cache_v + 0
    mx.eval(source_keys, source_values)
    growing_state = cached.new_state(512)
    baseline_state = cached.new_state(512)
    growing_state.cache = source
    baseline_state.cache = baseline
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2)
    previous = _unit(0.3)
    current = _current(0.7)
    noise = mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.15)
    kwargs = {
        "previous_unit": previous,
        "current_hidden": current,
        "speaker_condition": speaker,
        "steps": 2,
        "noise": noise,
    }
    if mode == "soar":
        kwargs.update(
            cfg_previous_unit=_unit(-0.3),
            cfg_current_hidden=_current(-0.7),
            guidance_scale=1.3,
        )

    expected = cached.sample_tail(baseline_state, **kwargs)
    actual = cached.sample_tail(growing_state, **kwargs)
    grown = growing_state.cache
    assert grown is not None
    mx.eval(expected, actual, grown.cache_k, grown.cache_v)

    np.testing.assert_array_equal(actual, expected)
    assert grown is not source
    assert grown.capacity_patches == target_capacity
    assert grown.offsets == [
        (source_capacity + 1) * _UNIT_LENGTH,
    ] * 2
    assert grown.cache_k.dtype == source.cache_k.dtype
    assert grown.cache_v.dtype == source.cache_v.dtype
    assert grown.cache_k.shape[:4] == source.cache_k.shape[:4]
    np.testing.assert_array_equal(
        grown.cache_k[..., : source.capacity_tokens, :],
        source_keys,
    )
    np.testing.assert_array_equal(
        grown.cache_v[..., : source.capacity_tokens, :],
        source_values,
    )
    assert baseline_state.cache is not None
    np.testing.assert_array_equal(
        grown.cache_k[..., : grown.valid_tokens, :],
        baseline_state.cache.cache_k[..., : baseline_state.cache.valid_tokens, :],
    )
    np.testing.assert_array_equal(
        grown.cache_v[..., : grown.valid_tokens, :],
        baseline_state.cache.cache_v[..., : baseline_state.cache.valid_tokens, :],
    )


@pytest.mark.parametrize("failure", ["allocation", "copy", "materialization"])
def test_cache_growth_failure_leaves_prior_cache_usable(
    monkeypatch,
    failure: str,
) -> None:
    _oracle, cached = _meanflow_pair()
    source = _published_cache(
        cached,
        capacity_patches=64,
        published_patches=64,
    )
    state = cached.new_state(512)
    state.cache = source
    original_keys = source.cache_k + 0
    original_values = source.cache_v + 0
    original_offsets = list(source.offsets)
    mx.eval(original_keys, original_values)

    if failure == "allocation":
        original = cached.runner.allocate_cache

        def fail(*args, **kwargs):
            raise RuntimeError("injected growth allocation failure")

        monkeypatch.setattr(cached.runner, "allocate_cache", fail)
    elif failure == "copy":
        original = cached._copy_published_cache

        def fail(*args, **kwargs):
            raise RuntimeError("injected growth copy failure")

        monkeypatch.setattr(cached, "_copy_published_cache", fail)
    else:
        original = cached._materialize_cache_growth

        def fail(*args, **kwargs):
            raise RuntimeError("injected growth materialization failure")

        monkeypatch.setattr(cached, "_materialize_cache_growth", fail)

    sample_kwargs = {
        "previous_unit": _unit(0.3),
        "current_hidden": _current(0.7),
        "speaker_condition": mx.full((1, _HIDDEN_SIZE), 0.2),
        "steps": 2,
        "noise": mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    }
    with pytest.raises(RuntimeError, match=f"growth {failure} failure"):
        cached.sample_tail(state, **sample_kwargs)

    assert state.cache is source
    assert source.capacity_patches == 64
    assert source.offsets == original_offsets
    np.testing.assert_array_equal(source.cache_k, original_keys)
    np.testing.assert_array_equal(source.cache_v, original_values)

    if failure == "allocation":
        monkeypatch.setattr(cached.runner, "allocate_cache", original)
    elif failure == "copy":
        monkeypatch.setattr(cached, "_copy_published_cache", original)
    else:
        monkeypatch.setattr(cached, "_materialize_cache_growth", original)
    cached.sample_tail(state, **sample_kwargs)
    assert state.cache is not source
    assert state.cache is not None
    assert state.cache.capacity_patches == 128


def test_cache_rejects_overflow_alignment_and_inconsistent_state() -> None:
    mx.random.seed(97)
    _oracle, cached = _meanflow_pair()
    with pytest.raises(ValueError, match="at most 512"):
        cached.new_state(513)
    state = cached.new_state(64)
    invalid = mx.zeros((1, _PATCH_SIZE + 2, _HIDDEN_SIZE))
    with pytest.raises(ValueError, match="unit-aligned"):
        cached.sample(state, sequence=invalid, steps=2)
    over_capacity = mx.zeros((1, 64 * _UNIT_LENGTH + 1 + _PATCH_SIZE, _HIDDEN_SIZE))
    with pytest.raises(ValueError, match="exceeds its capacity"):
        cached.sample(state, sequence=over_capacity, steps=2)

    sequence = _sequence([_unit(0.2)], _current(0.4))
    speaker = mx.zeros((1, _HIDDEN_SIZE))
    cached.sample(
        state,
        sequence=sequence,
        speaker_condition=speaker,
        steps=2,
        noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    )
    with pytest.raises(ValueError, match="offset"):
        cached.sample(
            state,
            sequence=sequence,
            speaker_condition=speaker,
            steps=2,
            noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
        )


def test_later_nfe_failure_keeps_existing_cache_transactional(monkeypatch) -> None:
    mx.random.seed(99)
    _oracle, cached = _meanflow_pair()
    state = cached.new_state(64)
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2)
    previous = _unit(0.3)
    second_sequence = _sequence([previous], _current(0.7))
    cached.sample(
        state,
        sequence=second_sequence,
        speaker_condition=speaker,
        steps=2,
        noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    )
    cache = state.cache
    assert cache is not None
    original_offsets = list(cache.offsets)
    original_keys = cache.cache_k + 0
    original_values = cache.cache_v + 0
    mx.eval(original_keys, original_values)
    third_sequence = _sequence([previous, _unit(0.9)], _current(1.3))
    original_next_velocity = cached.runner.next_velocity

    def fail_second_nfe(*args, **kwargs):
        if kwargs["nfe_index"] == 1:
            raise RuntimeError("injected later NFE failure")
        return original_next_velocity(*args, **kwargs)

    monkeypatch.setattr(cached.runner, "next_velocity", fail_second_nfe)
    with pytest.raises(RuntimeError, match="injected later NFE"):
        cached.sample(
            state,
            sequence=third_sequence,
            speaker_condition=speaker,
            steps=2,
            noise=mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 0.2,
        )
    assert cache.offsets == original_offsets
    np.testing.assert_array_equal(cache.cache_k, original_keys)
    np.testing.assert_array_equal(cache.cache_v, original_values)

    monkeypatch.setattr(cached.runner, "next_velocity", original_next_velocity)
    cached.sample(
        state,
        sequence=third_sequence,
        speaker_condition=speaker,
        steps=2,
        noise=mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 0.2,
    )
    assert cache.offsets == [2 * _UNIT_LENGTH, 2 * _UNIT_LENGTH]


def test_prompt_prefill_streams_nfe_and_publishes_only_after_success(
    monkeypatch,
) -> None:
    mx.random.seed(100)
    _oracle, cached = _meanflow_pair()
    state = cached.new_state(64)
    speaker = mx.full((1, _HIDDEN_SIZE), 0.15)
    sequence = _sequence([_unit(0.1), _unit(0.5), _unit(0.9)], _current(1.3))
    original_prefill_nfe = cached.runner.prefill_nfe
    prefill_calls = 0

    def fail_second_prefill(*args, **kwargs):
        nonlocal prefill_calls
        assert state.cache is None
        prefill_calls += 1
        if prefill_calls == 2:
            raise RuntimeError("injected prompt prefill failure")
        return original_prefill_nfe(*args, **kwargs)

    monkeypatch.setattr(cached.runner, "prefill_nfe", fail_second_prefill)
    with pytest.raises(RuntimeError, match="injected prompt prefill"):
        cached.sample(
            state,
            sequence=sequence,
            speaker_condition=speaker,
            steps=2,
            noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
        )
    assert prefill_calls == 2
    assert state.cache is None

    monkeypatch.setattr(cached.runner, "prefill_nfe", original_prefill_nfe)
    cached.sample(
        state,
        sequence=sequence,
        speaker_condition=speaker,
        steps=2,
        noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    )
    assert state.cache is not None
    assert state.cache.offsets == [3 * _UNIT_LENGTH, 3 * _UNIT_LENGTH]


def test_cache_write_overflow_and_runner_construction_preserve_weights() -> None:
    mx.random.seed(101)
    model, projection = _model_and_projection(meanflow=True)
    before = set(tree_flatten(model.parameters(), destination={}))
    cached = CachedMeanFlowSolver(
        model,
        projection,
        latent_dim=_LATENT_DIM,
        patch_size=_PATCH_SIZE,
    )
    after = set(tree_flatten(model.parameters(), destination={}))
    assert before == after
    cache = DiTKvCache.allocate(
        capacity_patches=64,
        unit_length=1,
        nfe=1,
        num_layers=1,
        branch_count=1,
        batch_size=1,
        num_heads=1,
        head_dim=1,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )
    with pytest.raises(ValueError, match="dtype"):
        cache.write(
            0,
            mx.ones((1, 1, 1, 1, 1), dtype=mx.bfloat16),
            mx.ones((1, 1, 1, 1, 1), dtype=mx.bfloat16),
        )
    full = mx.ones((1, 1, 1, 64, 1))
    cache.write(0, full, full)
    with pytest.raises(ValueError, match="overflow"):
        cache.write(0, mx.ones((1, 1, 1, 1, 1)), mx.ones((1, 1, 1, 1, 1)))
    state = cached.new_state(500)
    assert state.max_patches == 500
    assert state.capacity_patches == 512
