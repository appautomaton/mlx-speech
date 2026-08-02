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
        shape = (
            cache.num_layers,
            cache.branch_count * cache.batch_size,
            cache.num_heads,
            published_tokens,
            cache.head_dim,
        )
        size = int(np.prod(shape))
        values = mx.arange(size, dtype=mx.float32).reshape(shape)
        values = values / 200.0 + nfe_index / 10.0
        for layer_index in range(cache.num_layers):
            cache.cache_k[nfe_index][layer_index][..., :published_tokens, :] = (
                values[layer_index]
            )
            cache.cache_v[nfe_index][layer_index][..., :published_tokens, :] = (
                -values[layer_index]
            )
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
        key_dtype=source.key_dtype,
        value_dtype=source.value_dtype,
    )
    for nfe_index, offset in enumerate(source.offsets):
        for layer_index in range(source.num_layers):
            cache.cache_k[nfe_index][layer_index][..., :offset, :] = source.cache_k[
                nfe_index
            ][layer_index][..., :offset, :]
            cache.cache_v[nfe_index][layer_index][..., :offset, :] = source.cache_v[
                nfe_index
            ][layer_index][..., :offset, :]
    cache.offsets = list(source.offsets)
    mx.eval(cache.cache_k, cache.cache_v)
    return cache


def _cache_keys(cache: DiTKvCache) -> mx.array:
    return cache.stacked_keys()


def _cache_values(cache: DiTKvCache) -> mx.array:
    return cache.stacked_values()


def test_cache_scratch_storage_is_physical_per_nfe_layer() -> None:
    cache = DiTKvCache.allocate(
        capacity_patches=64,
        unit_length=1,
        nfe=2,
        num_layers=2,
        branch_count=1,
        batch_size=1,
        num_heads=1,
        head_dim=2,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )
    key_storage_ids = {
        id(layer) for nfe_layers in cache.cache_k for layer in nfe_layers
    }
    value_storage_ids = {
        id(layer) for nfe_layers in cache.cache_v for layer in nfe_layers
    }
    assert len(key_storage_ids) == len(value_storage_ids) == 4
    assert cache.keys.shape == cache.values.shape == cache.storage_shape

    keys = mx.full((1, 1, 2, 2), 3.0)
    values = mx.full((1, 1, 2, 2), -5.0)
    cache.write_scratch(1, 1, keys, values)
    mx.eval(cache.cache_k, cache.cache_v)

    assert cache.offsets == [0, 0]
    np.testing.assert_array_equal(cache.cache_k[1][1][..., :2, :], keys)
    np.testing.assert_array_equal(cache.cache_v[1][1][..., :2, :], values)
    for nfe_index, layer_index in ((0, 0), (0, 1), (1, 0)):
        np.testing.assert_array_equal(
            cache.cache_k[nfe_index][layer_index],
            mx.zeros_like(cache.cache_k[nfe_index][layer_index]),
        )
        np.testing.assert_array_equal(
            cache.cache_v[nfe_index][layer_index],
            mx.zeros_like(cache.cache_v[nfe_index][layer_index]),
        )


def test_cache_scratch_window_rejects_stale_writes_after_reopen_and_publish() -> None:
    cache = DiTKvCache.allocate(
        capacity_patches=64,
        unit_length=1,
        nfe=1,
        num_layers=1,
        branch_count=1,
        batch_size=1,
        num_heads=1,
        head_dim=2,
        key_dtype=mx.float32,
        value_dtype=mx.float32,
    )
    keys = mx.ones((1, 1, 2, 2), dtype=mx.float32)
    values = -keys
    first = cache.open_scratch_window()
    second = cache.open_scratch_window()
    with pytest.raises(RuntimeError, match="stale"):
        first.write(0, 0, keys, values)
    second.write(0, 0, keys, values)
    cache.publish_unit()
    with pytest.raises(RuntimeError, match="stale"):
        second.write(0, 0, keys, values)


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


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_tail_compile_signature_is_shared_across_layers_nfes_and_cache_buckets(
    mode: str,
) -> None:
    mx.random.seed(76)
    _oracle, cached = _meanflow_pair() if mode == "meanflow" else _soar_pair()
    speaker = mx.full((1, _HIDDEN_SIZE), 0.2)
    kwargs = {
        "previous_unit": _unit(0.3),
        "current_hidden": _current(0.7),
        "speaker_condition": speaker,
        "steps": 2,
        "noise": mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.15),
    }
    if mode == "soar":
        kwargs.update(
            cfg_previous_unit=_unit(-0.3),
            cfg_current_hidden=_current(-0.7),
            guidance_scale=1.3,
        )

    for capacity, published_patches in ((64, 1), (128, 5)):
        state = cached.new_state(512)
        state.cache = _published_cache(
            cached,
            capacity_patches=capacity,
            published_patches=published_patches,
        )
        output = cached.sample_tail(state, **kwargs)
        mx.eval(output)

    keys = tuple(cached.runner._compiled_tail_functions)
    assert len(keys) == 1
    assert keys[0].mode == mode
    assert keys[0].shape == (
        cached.runner.branch_count,
        2 * _UNIT_LENGTH,
        _HIDDEN_SIZE,
    )
    solver_type = CachedMeanFlowSolver if mode == "meanflow" else CachedSOARSolver
    next_request = solver_type(
        cached.runner.dit,
        cached.runner.coordinate_projection,
        latent_dim=_LATENT_DIM,
        patch_size=_PATCH_SIZE,
    )
    assert (
        next_request.runner._compiled_tail_functions
        is cached.runner._compiled_tail_functions
    )


def test_tail_builds_rotary_geometry_and_attention_bias_once_per_patch(
    monkeypatch,
) -> None:
    mx.random.seed(76)
    _oracle, cached = _meanflow_pair()
    state = cached.new_state(64)
    calls = {"rotary": 0, "bias": 0}
    original_rotary = cached.runner.prepare_rotary
    original_bias = cached.runner.prepare_attention_bias

    def count_rotary(*args, **kwargs):
        calls["rotary"] += 1
        return original_rotary(*args, **kwargs)

    def count_bias(*args, **kwargs):
        calls["bias"] += 1
        return original_bias(*args, **kwargs)

    monkeypatch.setattr(cached.runner, "prepare_rotary", count_rotary)
    monkeypatch.setattr(cached.runner, "prepare_attention_bias", count_bias)
    output = cached.sample_tail(
        state,
        previous_unit=_unit(0.3),
        current_hidden=_current(0.7),
        speaker_condition=mx.full((1, _HIDDEN_SIZE), 0.2),
        steps=2,
        noise=mx.full((1, _PATCH_SIZE, _LATENT_DIM), 0.15),
    )
    mx.eval(output)

    assert calls == {"rotary": 1, "bias": 1}


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_cached_interleaved_requests_are_exactly_isolated(mode: str) -> None:
    mx.random.seed(77)
    if mode == "meanflow":
        oracle, cached = _meanflow_pair()
    else:
        oracle, cached = _soar_pair()

    speakers = {
        "a": mx.full((1, _HIDDEN_SIZE), 0.11),
        "b": mx.full((1, _HIDDEN_SIZE), 0.37),
    }
    histories = {
        "a": [_unit(0.15), _unit(0.55), _unit(0.95)],
        "b": [_unit(1.4), _unit(1.8), _unit(2.2)],
    }
    cfg_histories = {
        "a": [_unit(-0.25), _unit(-0.65), _unit(-1.05)],
        "b": [_unit(-1.5), _unit(-1.9), _unit(-2.3)],
    }
    currents = {
        "a": [_current(0.35), _current(0.75), _current(1.15)],
        "b": [_current(1.6), _current(2.0), _current(2.4)],
    }
    cfg_currents = {
        "a": [_current(-0.45), _current(-0.85), _current(-1.25)],
        "b": [_current(-1.7), _current(-2.1), _current(-2.5)],
    }
    noises = {
        "a": [
            mx.full((1, _PATCH_SIZE, _LATENT_DIM), value)
            for value in (0.05, 0.15, 0.25)
        ],
        "b": [
            mx.full((1, _PATCH_SIZE, _LATENT_DIM), value)
            for value in (0.45, 0.55, 0.65)
        ],
    }

    def trusted_output(request: str, index: int) -> mx.array:
        sequence = _sequence(histories[request][: index + 1], currents[request][index])
        if mode == "meanflow":
            return _oracle_meanflow(
                oracle,
                sequence,
                noises[request][index],
                speakers[request],
            )
        return _oracle_soar(
            oracle,
            sequence,
            _sequence(
                cfg_histories[request][: index + 1],
                cfg_currents[request][index],
            ),
            noises[request][index],
            speakers[request],
        )

    def cached_output(state, request: str, index: int) -> mx.array:
        kwargs = {
            "sequence": _sequence(
                histories[request][: index + 1], currents[request][index]
            ),
            "speaker_condition": speakers[request],
            "steps": 2,
            "noise": noises[request][index],
        }
        if mode == "soar":
            kwargs.update(
                cfg_sequence=_sequence(
                    cfg_histories[request][: index + 1],
                    cfg_currents[request][index],
                ),
                guidance_scale=1.3,
            )
        return cached.sample(state, **kwargs)

    baseline_states = {request: cached.new_state(64) for request in ("a", "b")}
    baseline_outputs = {"a": [], "b": []}
    for request in ("a", "b"):
        for index in range(3):
            expected = trusted_output(request, index)
            actual = cached_output(baseline_states[request], request, index)
            mx.eval(expected, actual)
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
            saved = actual + 0
            mx.eval(saved)
            baseline_outputs[request].append(saved)

    interleaved_states = {request: cached.new_state(64) for request in ("a", "b")}
    for request, index in (
        ("a", 0),
        ("b", 0),
        ("a", 1),
        ("b", 1),
        ("a", 2),
        ("b", 2),
    ):
        other = "b" if request == "a" else "a"
        other_cache = interleaved_states[other].cache
        if other_cache is not None:
            other_offsets = list(other_cache.offsets)
            other_keys = _cache_keys(other_cache) + 0
            other_values = _cache_values(other_cache) + 0
            mx.eval(other_keys, other_values)

        expected = trusted_output(request, index)
        actual = cached_output(interleaved_states[request], request, index)
        mx.eval(expected, actual)
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(actual, baseline_outputs[request][index])

        if other_cache is not None:
            assert interleaved_states[other].cache is other_cache
            assert other_cache.offsets == other_offsets
            np.testing.assert_array_equal(_cache_keys(other_cache), other_keys)
            np.testing.assert_array_equal(_cache_values(other_cache), other_values)

    for request in ("a", "b"):
        cache = interleaved_states[request].cache
        baseline_cache = baseline_states[request].cache
        assert cache is not None and baseline_cache is not None
        assert cache.capacity_patches == baseline_cache.capacity_patches == 64
        assert cache.storage_shape == baseline_cache.storage_shape
        assert cache.offsets == baseline_cache.offsets == [3 * _UNIT_LENGTH] * 2
        np.testing.assert_array_equal(_cache_keys(cache), _cache_keys(baseline_cache))
        np.testing.assert_array_equal(
            _cache_values(cache), _cache_values(baseline_cache)
        )

    first_cache = interleaved_states["a"].cache
    second_cache = interleaved_states["b"].cache
    assert first_cache is not None and second_cache is not None
    first_storage = {
        id(layer)
        for storage in (first_cache.cache_k, first_cache.cache_v)
        for nfe_layers in storage
        for layer in nfe_layers
    }
    second_storage = {
        id(layer)
        for storage in (second_cache.cache_k, second_cache.cache_v)
        for nfe_layers in storage
        for layer in nfe_layers
    }
    assert first_cache is not second_cache
    assert first_storage.isdisjoint(second_storage)


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
            _cache_keys(cache)[0, :, :, :, : cache.valid_tokens],
            _cache_keys(cache)[1, :, :, :, : cache.valid_tokens],
        ).item()
    )
    scratch_end = cache.valid_tokens + _UNIT_LENGTH
    stacked_keys = _cache_keys(cache)
    assert bool(mx.any(stacked_keys[..., cache.valid_tokens : scratch_end, :] != 0))
    np.testing.assert_array_equal(
        stacked_keys[..., scratch_end:, :],
        mx.zeros_like(stacked_keys[..., scratch_end:, :]),
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
        _cache_keys(first_state.cache)[..., :3, :],
        _cache_keys(second_state.cache)[..., :3, :],
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        _cache_values(first_state.cache)[..., :3, :],
        _cache_values(second_state.cache)[..., :3, :],
        atol=2e-5,
        rtol=2e-5,
    )
    assert not bool(
        mx.allclose(
            _cache_keys(first_state.cache)[..., 3:6, :],
            _cache_keys(second_state.cache)[..., 3:6, :],
        ).item()
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
    assert cache.storage_shape[:3] == (2, 2, 2)
    assert cache.offsets == [3, 3]
    assert not bool(
        mx.allclose(
            _cache_keys(cache)[0, :, 0, :, :3],
            _cache_keys(cache)[0, :, 1, :, :3],
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
    assert state.cache.storage_shape[2] == 4
    assert state.cache.offsets == [_UNIT_LENGTH, _UNIT_LENGTH]


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_bfloat16_later_patch_uses_projected_cache_dtypes(
    monkeypatch,
    mode: str,
) -> None:
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
    projected_dtypes = []
    original_write_scratch = DiTKvCache._write_scratch_window

    def record_projected_dtypes(
        cache_self, window, nfe_index, layer_index, keys, values
    ):
        projected_dtypes.append((keys.dtype, values.dtype))
        return original_write_scratch(
            cache_self,
            window,
            nfe_index,
            layer_index,
            keys,
            values,
        )

    monkeypatch.setattr(
        DiTKvCache,
        "_write_scratch_window",
        record_projected_dtypes,
    )
    if mode == "meanflow":
        expected = _oracle_meanflow(oracle, sequence, noise, speaker)
        actual = cached.sample(
            state,
            sequence=sequence,
            speaker_condition=speaker,
            steps=2,
            noise=noise,
        )
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
    assert cache.key_dtype == cache.value_dtype == mx.bfloat16
    assert projected_dtypes
    assert all(key_dtype == cache.key_dtype for key_dtype, _ in projected_dtypes)
    assert all(
        value_dtype == cache.value_dtype for _, value_dtype in projected_dtypes
    )
    assert bool(mx.any(_cache_keys(cache)[..., :_UNIT_LENGTH, :] != 0).item())

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
    assert all(key_dtype == cache.key_dtype for key_dtype, _ in projected_dtypes)
    assert all(
        value_dtype == cache.value_dtype for _, value_dtype in projected_dtypes
    )


@pytest.mark.parametrize("mode", ["meanflow", "soar"])
def test_bfloat16_solver_input_hoists_invariant_projection_exactly(mode: str) -> None:
    mx.random.seed(941)
    model, projection = _model_and_projection(meanflow=mode == "meanflow")
    model.set_dtype(mx.bfloat16)
    projection.set_dtype(mx.bfloat16)
    solver = (
        CachedMeanFlowSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        )
        if mode == "meanflow"
        else CachedSOARSolver(
            model,
            projection,
            latent_dim=_LATENT_DIM,
            patch_size=_PATCH_SIZE,
        )
    )
    invariant = mx.random.normal((1, 2 * _UNIT_LENGTH - _PATCH_SIZE, _HIDDEN_SIZE))
    invariant = invariant.astype(mx.bfloat16)
    cfg_invariant = (
        None
        if mode == "meanflow"
        else mx.random.normal(invariant.shape).astype(mx.bfloat16)
    )
    coordinate = mx.random.normal((1, _PATCH_SIZE, _LATENT_DIM)).astype(mx.bfloat16)
    projected_coordinate = projection(coordinate).astype(mx.bfloat16)
    conditional = mx.concatenate((invariant, projected_coordinate), axis=1)
    if cfg_invariant is None:
        branches = conditional
    else:
        branches = mx.concatenate(
            (
                conditional,
                mx.concatenate((cfg_invariant, projected_coordinate), axis=1),
            ),
            axis=0,
        )
    expected = model.input_layer(branches)
    projected_invariant = solver.runner.project_invariant_input(
        invariant,
        cfg_invariant,
    )
    actual = solver.runner._compose_solver_input(
        coordinate,
        invariant_input=invariant,
        cfg_invariant_input=cfg_invariant,
        projected_invariant=projected_invariant,
    )
    mx.eval(expected, actual)
    np.testing.assert_array_equal(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
    )


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
    source_keys = _cache_keys(source) + 0
    source_values = _cache_values(source) + 0
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
    assert grown.key_dtype == source.key_dtype
    assert grown.value_dtype == source.value_dtype
    assert grown.storage_shape[:4] == source.storage_shape[:4]
    np.testing.assert_array_equal(
        _cache_keys(grown)[..., : source.capacity_tokens, :],
        source_keys,
    )
    np.testing.assert_array_equal(
        _cache_values(grown)[..., : source.capacity_tokens, :],
        source_values,
    )
    assert baseline_state.cache is not None
    np.testing.assert_array_equal(
        _cache_keys(grown)[..., : grown.valid_tokens, :],
        _cache_keys(baseline_state.cache)[
            ..., : baseline_state.cache.valid_tokens, :
        ],
    )
    np.testing.assert_array_equal(
        _cache_values(grown)[..., : grown.valid_tokens, :],
        _cache_values(baseline_state.cache)[
            ..., : baseline_state.cache.valid_tokens, :
        ],
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
    original_keys = _cache_keys(source) + 0
    original_values = _cache_values(source) + 0
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
    np.testing.assert_array_equal(_cache_keys(source), original_keys)
    np.testing.assert_array_equal(_cache_values(source), original_values)

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


def test_mid_nfe_failure_preserves_exact_published_prefix_and_retry(monkeypatch) -> None:
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
    published_length = cache.valid_tokens
    original_keys = _cache_keys(cache)[..., :published_length, :] + 0
    original_values = _cache_values(cache)[..., :published_length, :] + 0
    mx.eval(original_keys, original_values)
    third_sequence = _sequence([previous, _unit(0.9)], _current(1.3))
    third_noise = mx.ones((1, _PATCH_SIZE, _LATENT_DIM)) * 0.2
    clean_state = cached.new_state(64)
    cached.sample(
        clean_state,
        sequence=second_sequence,
        speaker_condition=speaker,
        steps=2,
        noise=mx.zeros((1, _PATCH_SIZE, _LATENT_DIM)),
    )
    expected = cached.sample(
        clean_state,
        sequence=third_sequence,
        speaker_condition=speaker,
        steps=2,
        noise=third_noise,
    )
    assert clean_state.cache is not None
    original_write_scratch = DiTKvCache._write_scratch_window

    def fail_mid_nfe(cache_self, window, nfe_index, layer_index, keys, values):
        end = original_write_scratch(
            cache_self,
            window,
            nfe_index,
            layer_index,
            keys,
            values,
        )
        if nfe_index == 1 and layer_index == 0:
            raise RuntimeError("injected mid-NFE failure")
        return end

    monkeypatch.setattr(DiTKvCache, "_write_scratch_window", fail_mid_nfe)
    with pytest.raises(RuntimeError, match="injected mid-NFE"):
        cached.sample(
            state,
            sequence=third_sequence,
            speaker_condition=speaker,
            steps=2,
            noise=third_noise,
        )
    assert cache.offsets == original_offsets
    np.testing.assert_array_equal(
        _cache_keys(cache)[..., :published_length, :], original_keys
    )
    np.testing.assert_array_equal(
        _cache_values(cache)[..., :published_length, :], original_values
    )

    scratch_end = published_length + 2 * _UNIT_LENGTH
    for nfe_layers in cache.cache_k:
        for layer in nfe_layers:
            layer[..., published_length:scratch_end, :] = 37.0
    for nfe_layers in cache.cache_v:
        for layer in nfe_layers:
            layer[..., published_length:scratch_end, :] = -41.0
    mx.eval(cache.cache_k, cache.cache_v)
    monkeypatch.setattr(
        DiTKvCache,
        "_write_scratch_window",
        original_write_scratch,
    )
    actual = cached.sample(
        state,
        sequence=third_sequence,
        speaker_condition=speaker,
        steps=2,
        noise=third_noise,
    )
    assert cache.offsets == [2 * _UNIT_LENGTH, 2 * _UNIT_LENGTH]
    mx.eval(expected, actual, cache.cache_k, cache.cache_v)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        _cache_keys(cache)[..., : cache.valid_tokens, :],
        _cache_keys(clean_state.cache)[..., : clean_state.cache.valid_tokens, :],
    )
    np.testing.assert_array_equal(
        _cache_values(cache)[..., : cache.valid_tokens, :],
        _cache_values(clean_state.cache)[..., : clean_state.cache.valid_tokens, :],
    )
    np.testing.assert_array_equal(
        _cache_keys(cache)[..., :published_length, :], original_keys
    )
    np.testing.assert_array_equal(
        _cache_values(cache)[..., :published_length, :], original_values
    )


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
