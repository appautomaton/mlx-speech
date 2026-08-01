"""Inference-only delayed-commit DiT cache for dots.tts solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from .dit import DiT, modulate
from .solvers import ODESchedule, build_ode_schedule


DIT_CACHE_BUCKETS = (64, 128, 256, 512)
SolverMode = Literal["meanflow", "soar"]
BlockModulation = tuple[mx.array, ...]
PreparedModulations = tuple[
    tuple[BlockModulation, ...],
    tuple[mx.array, mx.array],
]
CacheFactory = Callable[[mx.Dtype, mx.Dtype], "DiTKvCache"]


def resolve_dit_cache_bucket(patch_count: int) -> int:
    """Resolve a request patch bound to one of the official cache buckets."""

    if isinstance(patch_count, bool) or int(patch_count) <= 0:
        raise ValueError("DiT cache patch count must be positive")
    requested = int(patch_count)
    for bucket in DIT_CACHE_BUCKETS:
        if requested <= bucket:
            return bucket
    raise ValueError(
        f"dots.tts DiT cache supports at most 512 patches: requested={requested}"
    )


@dataclass
class DiTKvCache:
    """Per-NFE/layer/CFG-branch bounded K/V storage for one request."""

    capacity_patches: int
    unit_length: int
    nfe: int
    num_layers: int
    branch_count: int
    batch_size: int
    cache_k: tuple[tuple[mx.array, ...], ...]
    cache_v: tuple[tuple[mx.array, ...], ...]
    offsets: list[int]

    def __post_init__(self) -> None:
        if self.capacity_patches not in DIT_CACHE_BUCKETS:
            raise ValueError(
                "DiT cache capacity must use an official 64/128/256/512 bucket"
            )
        if (
            min(
                self.unit_length,
                self.nfe,
                self.num_layers,
                self.branch_count,
                self.batch_size,
            )
            <= 0
        ):
            raise ValueError("DiT cache dimensions must be positive")
        if len(self.cache_k) != self.nfe or len(self.cache_v) != self.nfe:
            raise ValueError(
                "DiT cache storage must contain one layer table per NFE"
            )
        if any(len(layers) != self.num_layers for layers in self.cache_k) or any(
            len(layers) != self.num_layers for layers in self.cache_v
        ):
            raise ValueError("DiT cache storage must contain every model layer")
        expected_shape = (
            self.branch_count * self.batch_size,
            self.num_heads,
            self.capacity_tokens,
            self.head_dim,
        )
        key_dtype = self.cache_k[0][0].dtype
        value_dtype = self.cache_v[0][0].dtype
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("DiT cache head dimensions must be positive")
        for nfe_index in range(self.nfe):
            for layer_index in range(self.num_layers):
                keys = self.cache_k[nfe_index][layer_index]
                values = self.cache_v[nfe_index][layer_index]
                if keys.ndim != 4 or values.ndim != 4:
                    raise ValueError(
                        "DiT cache layer arrays must have shape "
                        "(branch_batch, head, token, head_dim)"
                    )
                if keys.shape != expected_shape or values.shape != expected_shape:
                    raise ValueError(
                        "DiT cache storage does not match its layer metadata"
                    )
                if keys.dtype != key_dtype or values.dtype != value_dtype:
                    raise ValueError("DiT cache layer dtypes must be uniform")
        if len(self.offsets) != self.nfe:
            raise ValueError("DiT cache must track one offset per NFE")
        if any(offset < 0 or offset > self.capacity_tokens for offset in self.offsets):
            raise ValueError("DiT cache offset is outside its capacity")
        if any(offset % self.unit_length for offset in self.offsets):
            raise ValueError("DiT cache offsets must be unit-aligned")

    @property
    def capacity_tokens(self) -> int:
        return self.capacity_patches * self.unit_length

    @property
    def valid_tokens(self) -> int:
        valid = set(self.offsets)
        if len(valid) != 1:
            raise RuntimeError(f"DiT cache NFE offsets diverged: {self.offsets}")
        return self.offsets[0]

    @property
    def num_heads(self) -> int:
        return int(self.cache_k[0][0].shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.cache_k[0][0].shape[-1])

    @property
    def key_dtype(self) -> mx.Dtype:
        return self.cache_k[0][0].dtype

    @property
    def value_dtype(self) -> mx.Dtype:
        return self.cache_v[0][0].dtype

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return (
            self.nfe,
            self.num_layers,
            self.branch_count * self.batch_size,
            self.num_heads,
            self.capacity_tokens,
            self.head_dim,
        )

    def stacked_keys(self) -> mx.array:
        """Materialize a unified inspection view without using it for inference."""

        return mx.stack(
            tuple(mx.stack(layers, axis=0) for layers in self.cache_k),
            axis=0,
        )

    def stacked_values(self) -> mx.array:
        """Materialize a unified inspection view without using it for inference."""

        return mx.stack(
            tuple(mx.stack(layers, axis=0) for layers in self.cache_v),
            axis=0,
        )

    @property
    def keys(self) -> mx.array:
        return self.stacked_keys()

    @property
    def values(self) -> mx.array:
        return self.stacked_values()

    @classmethod
    def allocate(
        cls,
        *,
        capacity_patches: int,
        unit_length: int,
        nfe: int,
        num_layers: int,
        branch_count: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        key_dtype: mx.Dtype,
        value_dtype: mx.Dtype,
    ) -> DiTKvCache:
        capacity_tokens = int(capacity_patches) * int(unit_length)
        layer_shape = (
            int(branch_count) * int(batch_size),
            int(num_heads),
            capacity_tokens,
            int(head_dim),
        )
        cache_k = tuple(
            tuple(
                mx.zeros(layer_shape, dtype=key_dtype)
                for _layer_index in range(int(num_layers))
            )
            for _nfe_index in range(int(nfe))
        )
        cache_v = tuple(
            tuple(
                mx.zeros(layer_shape, dtype=value_dtype)
                for _layer_index in range(int(num_layers))
            )
            for _nfe_index in range(int(nfe))
        )
        return cls(
            capacity_patches=int(capacity_patches),
            unit_length=int(unit_length),
            nfe=int(nfe),
            num_layers=int(num_layers),
            branch_count=int(branch_count),
            batch_size=int(batch_size),
            cache_k=cache_k,
            cache_v=cache_v,
            offsets=[0] * int(nfe),
        )

    def write(
        self,
        nfe_index: int,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        """Append one independently computed prefix segment for one NFE."""

        start, end = self.validate_write(nfe_index, keys, values)
        for layer_index in range(self.num_layers):
            self.cache_k[nfe_index][layer_index][..., start:end, :] = keys[
                layer_index
            ]
            self.cache_v[nfe_index][layer_index][..., start:end, :] = values[
                layer_index
            ]
        self.offsets[nfe_index] = end

    def validate_write(
        self,
        nfe_index: int,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[int, int]:
        """Validate an append without changing cache contents or offsets."""

        if nfe_index < 0 or nfe_index >= self.nfe:
            raise ValueError(f"DiT cache NFE index is out of range: {nfe_index}")
        if keys.ndim != 5 or values.ndim != 5 or keys.shape != values.shape:
            raise ValueError(
                "DiT cache writes require matching "
                "(layer, branch_batch, head, token, head_dim) arrays"
            )
        expected = (
            self.num_layers,
            self.branch_count * self.batch_size,
            self.num_heads,
            self.head_dim,
        )
        actual = keys.shape[:3] + keys.shape[-1:]
        if actual != expected:
            raise ValueError(
                f"DiT cache write shape differs from storage: {keys.shape}"
            )
        if keys.dtype != self.key_dtype or values.dtype != self.value_dtype:
            raise ValueError("DiT cache write dtype differs from storage")
        start = self.offsets[nfe_index]
        write_length = int(keys.shape[-2])
        if write_length <= 0 or write_length % self.unit_length:
            raise ValueError("DiT cache writes must contain complete units")
        end = start + write_length
        if end > self.capacity_tokens:
            raise ValueError(
                f"DiT cache overflow: required={end} capacity={self.capacity_tokens}"
            )
        return start, end

    def write_scratch(
        self,
        nfe_index: int,
        layer_index: int,
        keys: mx.array,
        values: mx.array,
    ) -> int:
        """Write a fresh two-unit tail without publishing either unit."""

        if nfe_index < 0 or nfe_index >= self.nfe:
            raise ValueError(f"DiT cache NFE index is out of range: {nfe_index}")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(f"DiT cache layer index is out of range: {layer_index}")
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError(
                "DiT cache scratch writes require matching "
                "(branch_batch, head, token, head_dim) arrays"
            )
        expected = (
            self.branch_count * self.batch_size,
            self.num_heads,
            self.head_dim,
        )
        actual = keys.shape[:2] + keys.shape[-1:]
        if actual != expected:
            raise ValueError(
                f"DiT cache scratch shape differs from storage: {keys.shape}"
            )
        if keys.dtype != self.key_dtype or values.dtype != self.value_dtype:
            raise ValueError("DiT cache scratch dtype differs from storage")
        start = self.offsets[nfe_index]
        write_length = int(keys.shape[-2])
        if write_length != 2 * self.unit_length:
            raise ValueError("DiT cache scratch must contain exactly two units")
        end = start + write_length
        if end > self.capacity_tokens:
            raise ValueError(
                "DiT cache scratch overflow: "
                f"required={end} capacity={self.capacity_tokens}"
            )
        self.cache_k[nfe_index][layer_index][..., start:end, :] = keys
        self.cache_v[nfe_index][layer_index][..., start:end, :] = values
        return end

    def publish_unit(self) -> None:
        """Atomically expose one finalized unit for every NFE."""

        start = self.valid_tokens
        end = start + self.unit_length
        if end > self.capacity_tokens:
            raise ValueError(
                f"DiT cache overflow: required={end} capacity={self.capacity_tokens}"
            )
        self.offsets = [end] * self.nfe


@dataclass
class DiTSolverState:
    """Mutable request-local state for one cached DiT generation."""

    max_patches: int
    cache: DiTKvCache | None = None
    schedule: ODESchedule | None = None
    modulations_by_nfe: tuple[PreparedModulations, ...] | None = None
    mode: SolverMode | None = None
    batch_size: int | None = None
    activation_dtype: str | None = None
    _speaker_identity: int | None = field(default=None, repr=False)
    _condition_bound: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        resolve_dit_cache_bucket(self.max_patches)

    @property
    def capacity_patches(self) -> int:
        """Compatibility view of the bucket bounding this request maximum."""

        return resolve_dit_cache_bucket(self.max_patches)


def _mask_slice(
    mask: mx.array,
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> mx.array:
    if mask.ndim == 2:
        return mask[query_start:query_end, :key_end]
    if mask.ndim in {3, 4}:
        return mask[..., query_start:query_end, :key_end]
    raise ValueError("DiT attention mask must have rank 2-4")


def _causal_mask(length: int) -> mx.array:
    positions = mx.arange(length)
    return (positions[:, None] >= positions[None, :])[None]


def _delayed_tail_mask(persistent_length: int, unit_length: int) -> mx.array:
    causal = mx.arange(unit_length)[:, None] >= mx.arange(unit_length)[None, :]
    previous = mx.concatenate(
        (causal, mx.zeros((unit_length, unit_length), dtype=mx.bool_)), axis=1
    )
    current = mx.ones((unit_length, 2 * unit_length), dtype=mx.bool_)
    fresh = mx.concatenate((previous, current), axis=0)
    if persistent_length:
        prefix = mx.ones((2 * unit_length, persistent_length), dtype=mx.bool_)
        fresh = mx.concatenate((prefix, fresh), axis=1)
    return fresh[None]


def _enforce_required_mask(mask: mx.array, required: mx.array) -> mx.array:
    """Intersect a caller mask with a cache-correct causal boundary."""

    if mask.ndim == 2:
        required = required[0]
    elif mask.ndim == 4:
        required = required[:, None]
    if mask.dtype == mx.bool_:
        return mx.logical_and(mask, required)
    return mx.where(required, mask, float("-inf")).astype(mask.dtype)


def _expand_branch_positions(
    positions: mx.array,
    *,
    batch_size: int,
    branch_count: int,
) -> mx.array:
    if branch_count == 2 and positions.ndim == 2 and positions.shape[0] == batch_size:
        return mx.concatenate((positions, positions), axis=0)
    return positions


def _expand_branch_mask(
    mask: mx.array | None,
    *,
    batch_size: int,
    branch_count: int,
) -> mx.array | None:
    if (
        mask is not None
        and branch_count == 2
        and mask.ndim in {3, 4}
        and mask.shape[0] == batch_size
    ):
        return mx.concatenate((mask, mask), axis=0)
    return mask


class CachedDiTRunner:
    """Stateless tail runner that consumes request-owned K/V and modulations."""

    def __init__(
        self,
        dit: DiT,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
        patch_size: int,
        hidden_patch_size: int,
        mode: SolverMode,
    ):
        if patch_size <= 0 or hidden_patch_size <= 0 or latent_dim <= 0:
            raise ValueError("cached DiT patch and latent dimensions must be positive")
        if not dit.blocks:
            raise ValueError("cached DiT inference requires at least one block")
        if mode == "meanflow" and dit.duration_embedder is None:
            raise ValueError("cached MeanFlow requires a duration-enabled DiT")
        if mode == "soar" and dit.duration_embedder is not None:
            raise ValueError("cached SOAR requires a DiT without duration embedding")
        self.dit = dit
        self.coordinate_projection = coordinate_projection
        self.latent_dim = int(latent_dim)
        self.patch_size = int(patch_size)
        self.hidden_patch_size = int(hidden_patch_size)
        self.unit_length = self.hidden_patch_size + self.patch_size
        self.mode = mode
        self.branch_count = 1 if mode == "meanflow" else 2
        attention = dit.blocks[0].attn
        self.num_layers = len(dit.blocks)
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim

    def allocate_cache(
        self,
        *,
        capacity_patches: int,
        nfe: int,
        batch_size: int,
        key_dtype: mx.Dtype,
        value_dtype: mx.Dtype,
    ) -> DiTKvCache:
        return DiTKvCache.allocate(
            capacity_patches=capacity_patches,
            unit_length=self.unit_length,
            nfe=nfe,
            num_layers=self.num_layers,
            branch_count=self.branch_count,
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            key_dtype=key_dtype,
            value_dtype=value_dtype,
        )

    def _project_coordinate(self, coordinate: mx.array) -> mx.array:
        if coordinate.ndim != 3 or int(coordinate.shape[-1]) != self.latent_dim:
            raise ValueError(
                "cached DiT coordinate must have shape "
                f"(batch, patch, {self.latent_dim})"
            )
        projected = self.coordinate_projection(coordinate).astype(coordinate.dtype)
        expected = (
            coordinate.shape[0],
            self.patch_size,
            self.dit.hidden_size,
        )
        if projected.shape != expected:
            raise ValueError(
                "cached DiT coordinate projection does not match hidden size"
            )
        return projected

    def _run_blocks(
        self,
        sequence: mx.array,
        modulations: PreparedModulations,
        *,
        positions: mx.array,
        attention_mask: mx.array | None,
        cache: DiTKvCache | None = None,
        cache_factory: CacheFactory | None = None,
        nfe_index: int | None = None,
        commit_length: int = 0,
        scratch_length: int = 0,
    ) -> tuple[
        mx.array,
        mx.array | None,
        mx.array | None,
        DiTKvCache | None,
    ]:
        block_modulations, _final_modulation = modulations
        if len(block_modulations) != self.num_layers:
            raise ValueError("cached DiT modulation layer count differs from model")
        if cache is not None or cache_factory is not None:
            if nfe_index is None:
                raise ValueError("cached DiT attention requires an NFE index")
        value = self.dit.input_layer(sequence)
        committed_keys: list[mx.array] = []
        committed_values: list[mx.array] = []
        for layer_index, (block, block_modulation) in enumerate(
            zip(self.dit.blocks, block_modulations, strict=True)
        ):
            (
                shift_attn,
                scale_attn,
                gate_attn,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = block_modulation
            attention_input = modulate(block.norm1(value), shift_attn, scale_attn)
            query, key, projected_value = block.attn.project(
                attention_input, positions=positions
            )
            if commit_length:
                committed_keys.append(key[..., :commit_length, :])
                committed_values.append(projected_value[..., :commit_length, :])
            if scratch_length:
                if cache is None:
                    if cache_factory is None:
                        raise RuntimeError("cached DiT scratch has no storage factory")
                    cache = cache_factory(key.dtype, projected_value.dtype)
                if int(key.shape[-2]) != scratch_length:
                    raise ValueError("cached DiT projected scratch length differs")
                key_end = cache.write_scratch(
                    nfe_index,
                    layer_index,
                    key,
                    projected_value,
                )
                key = cache.cache_k[nfe_index][layer_index][..., :key_end, :]
                projected_value = cache.cache_v[nfe_index][layer_index][
                    ..., :key_end, :
                ]
            attended = block.attn.attend(
                query, key, projected_value, mask=attention_mask
            )
            value = value + gate_attn[:, None] * attended
            feed_forward = block.ffn(modulate(block.norm2(value), shift_ffn, scale_ffn))
            value = value + gate_ffn[:, None] * feed_forward
        if not commit_length:
            return value, None, None, cache
        return (
            value,
            mx.stack(committed_keys, axis=0),
            mx.stack(committed_values, axis=0),
            cache,
        )

    def _apply_final(
        self,
        value: mx.array,
        modulations: PreparedModulations,
    ) -> mx.array:
        shift, scale = modulations[1]
        return self.dit.output_layer.linear(
            modulate(self.dit.output_layer.norm(value), shift, scale)
        )

    def prefill_nfe(
        self,
        *,
        prefix_sequence: mx.array,
        modulations: PreparedModulations,
        cfg_prefix_sequence: mx.array | None,
        positions: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Project one prompt prefix NFE without retaining another NFE's K/V."""

        prefix_length = int(prefix_sequence.shape[1])
        if prefix_length == 0:
            raise ValueError("DiT prompt prefill requires a non-empty prefix")
        if self.mode == "soar":
            if cfg_prefix_sequence is None:
                raise ValueError("cached SOAR prefill requires a CFG prefix")
            branches = mx.concatenate((prefix_sequence, cfg_prefix_sequence), axis=0)
        else:
            if cfg_prefix_sequence is not None:
                raise ValueError("cached MeanFlow prefill cannot use a CFG prefix")
            branches = prefix_sequence
        _value, keys, values, _cache = self._run_blocks(
            branches,
            modulations,
            positions=positions,
            attention_mask=attention_mask,
            commit_length=prefix_length,
        )
        if keys is None or values is None:
            raise RuntimeError("DiT prefill did not produce K/V")
        return keys, values

    def first_velocity(
        self,
        coordinate: mx.array,
        *,
        current_hidden: mx.array,
        cfg_current_hidden: mx.array | None,
        modulations: PreparedModulations,
        positions: mx.array,
        attention_mask: mx.array | None,
        guidance_scale: float,
    ) -> mx.array:
        projected = self._project_coordinate(coordinate)
        conditional = mx.concatenate((current_hidden, projected), axis=1)
        if self.mode == "soar":
            if cfg_current_hidden is None:
                raise ValueError("cached SOAR first patch requires CFG hidden state")
            unconditional = mx.concatenate((cfg_current_hidden, projected), axis=1)
            branches = mx.concatenate((conditional, unconditional), axis=0)
        else:
            branches = conditional
        value, _keys, _values, _cache = self._run_blocks(
            branches,
            modulations,
            positions=positions,
            attention_mask=attention_mask,
        )
        prediction = self._apply_final(value, modulations)[:, self.hidden_patch_size :]
        if self.mode == "meanflow":
            return prediction
        batch_size = int(current_hidden.shape[0])
        conditional_velocity = prediction[:batch_size]
        unconditional_velocity = prediction[batch_size:]
        return conditional_velocity + float(guidance_scale) * (
            conditional_velocity - unconditional_velocity
        )

    def next_velocity(
        self,
        coordinate: mx.array,
        *,
        previous_unit: mx.array,
        current_hidden: mx.array,
        cfg_previous_unit: mx.array | None,
        cfg_current_hidden: mx.array | None,
        cache: DiTKvCache | None,
        cache_factory: CacheFactory | None,
        nfe_index: int,
        modulations: PreparedModulations,
        positions: mx.array,
        attention_mask: mx.array,
        guidance_scale: float,
    ) -> tuple[mx.array, DiTKvCache]:
        projected = self._project_coordinate(coordinate)
        conditional = mx.concatenate((previous_unit, current_hidden, projected), axis=1)
        if self.mode == "soar":
            if cfg_previous_unit is None or cfg_current_hidden is None:
                raise ValueError("cached SOAR tail requires both CFG units")
            unconditional = mx.concatenate(
                (cfg_previous_unit, cfg_current_hidden, projected), axis=1
            )
            branches = mx.concatenate((conditional, unconditional), axis=0)
        else:
            branches = conditional
        value, _keys, _values, cache = self._run_blocks(
            branches,
            modulations,
            positions=positions,
            attention_mask=attention_mask,
            cache=cache,
            cache_factory=cache_factory,
            nfe_index=nfe_index,
            scratch_length=2 * self.unit_length,
        )
        if cache is None:
            raise RuntimeError("cached DiT tail did not produce scratch K/V")
        latent_start = self.unit_length + self.hidden_patch_size
        prediction = self._apply_final(value, modulations)[:, latent_start:]
        if self.mode == "meanflow":
            velocity = prediction
        else:
            batch_size = int(current_hidden.shape[0])
            conditional_velocity = prediction[:batch_size]
            unconditional_velocity = prediction[batch_size:]
            velocity = conditional_velocity + float(guidance_scale) * (
                conditional_velocity - unconditional_velocity
            )
        return velocity, cache


class CachedDiTSolver:
    """Fixed-step MeanFlow/SOAR solver with delayed per-NFE K/V commits."""

    def __init__(
        self,
        dit: DiT,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
        patch_size: int = 4,
        hidden_patch_size: int = 1,
        mode: SolverMode,
    ):
        self.runner = CachedDiTRunner(
            dit,
            coordinate_projection,
            latent_dim=latent_dim,
            patch_size=patch_size,
            hidden_patch_size=hidden_patch_size,
            mode=mode,
        )
        self.mode = mode
        self.latent_dim = int(latent_dim)
        self.patch_size = int(patch_size)
        self.hidden_patch_size = int(hidden_patch_size)
        self.unit_length = self.runner.unit_length

    def new_state(self, max_patches: int) -> DiTSolverState:
        resolve_dit_cache_bucket(max_patches)
        return DiTSolverState(max_patches=int(max_patches))

    def _validate_state_cache(
        self,
        state: DiTSolverState,
        *,
        nfe: int,
        batch_size: int,
    ) -> None:
        cache = state.cache
        if cache is None:
            return
        if (
            cache.capacity_patches > state.capacity_patches
            or cache.unit_length != self.unit_length
            or cache.nfe != nfe
            or cache.num_layers != self.runner.num_layers
            or cache.branch_count != self.runner.branch_count
            or cache.batch_size != batch_size
        ):
            raise ValueError("DiT solver state cache is incompatible with this request")

    def _allocate_projected_cache(
        self,
        state: DiTSolverState,
        *,
        nfe: int,
        batch_size: int,
        keys: mx.array,
        values: mx.array,
    ) -> DiTKvCache:
        if state.cache is not None:
            raise RuntimeError("DiT solver state cache is already allocated")
        return self.runner.allocate_cache(
            capacity_patches=DIT_CACHE_BUCKETS[0],
            nfe=nfe,
            batch_size=batch_size,
            key_dtype=keys.dtype,
            value_dtype=values.dtype,
        )

    @staticmethod
    def _copy_published_cache(
        source: DiTKvCache,
        replacement: DiTKvCache,
    ) -> None:
        """Copy only offsets already published by every NFE."""

        for nfe_index, offset in enumerate(source.offsets):
            if offset:
                for layer_index in range(source.num_layers):
                    replacement.cache_k[nfe_index][layer_index][..., :offset, :] = (
                        source.cache_k[nfe_index][layer_index][..., :offset, :]
                    )
                    replacement.cache_v[nfe_index][layer_index][..., :offset, :] = (
                        source.cache_v[nfe_index][layer_index][..., :offset, :]
                    )
        replacement.offsets = list(source.offsets)

    @staticmethod
    def _materialize_cache_growth(cache: DiTKvCache) -> None:
        mx.eval(cache.cache_k, cache.cache_v)

    def _grow_projected_cache(
        self,
        cache: DiTKvCache,
        *,
        required_tokens: int,
    ) -> DiTKvCache:
        """Return a fully materialized larger cache without mutating the source."""

        if required_tokens <= cache.capacity_tokens:
            return cache
        required_patches = (
            required_tokens + self.unit_length - 1
        ) // self.unit_length
        capacity_patches = resolve_dit_cache_bucket(required_patches)
        replacement = self.runner.allocate_cache(
            capacity_patches=capacity_patches,
            nfe=cache.nfe,
            batch_size=cache.batch_size,
            key_dtype=cache.key_dtype,
            value_dtype=cache.value_dtype,
        )
        self._copy_published_cache(cache, replacement)
        self._materialize_cache_growth(replacement)
        return replacement

    def _prepare_request(
        self,
        state: DiTSolverState,
        *,
        speaker_condition: mx.array | None,
        steps: int | None,
        batch_size: int,
        dtype: mx.Dtype,
    ) -> tuple[ODESchedule, tuple[PreparedModulations, ...]]:
        schedule = build_ode_schedule(self.mode, steps, dtype)
        nfe = int(schedule.times.shape[0])
        speaker_identity = None if speaker_condition is None else id(speaker_condition)
        if state._condition_bound:
            if (
                state.mode != self.mode
                or state.batch_size != batch_size
                or state.activation_dtype != str(dtype)
                or state.schedule is None
                or int(state.schedule.times.shape[0]) != nfe
                or state._speaker_identity != speaker_identity
            ):
                raise ValueError(
                    "DiT solver state was already bound to different request conditioning"
                )
            if state.modulations_by_nfe is None:
                raise RuntimeError("DiT solver state has no prepared modulations")
            return state.schedule, state.modulations_by_nfe
        if speaker_condition is not None and speaker_condition.shape != (
            batch_size,
            self.runner.dit.hidden_size,
        ):
            raise ValueError(
                "speaker condition must match the DiT batch and hidden size"
            )
        if self.mode == "soar" and speaker_condition is not None:
            branch_speaker = mx.concatenate(
                (speaker_condition, mx.zeros_like(speaker_condition)), axis=0
            )
        else:
            branch_speaker = speaker_condition
        branch_batch = batch_size * self.runner.branch_count
        prepared: list[PreparedModulations] = []
        for nfe_index in range(nfe):
            timestep = mx.broadcast_to(
                schedule.times[nfe_index : nfe_index + 1], (branch_batch,)
            )
            duration = None
            if self.mode == "meanflow":
                duration = mx.full((branch_batch,), schedule.step_size, dtype=dtype)
            condition = self.runner.dit.prepare_condition(
                timestep,
                duration=duration,
                speaker_condition=branch_speaker,
            )
            prepared.append(self.runner.dit.prepare_modulations(condition))
        state.schedule = schedule
        state.modulations_by_nfe = tuple(prepared)
        state.mode = self.mode
        state.batch_size = batch_size
        state.activation_dtype = str(dtype)
        state._speaker_identity = speaker_identity
        state._condition_bound = True
        return schedule, state.modulations_by_nfe

    def _validate_inputs(
        self,
        state: DiTSolverState,
        *,
        sequence: mx.array,
        cfg_sequence: mx.array | None,
        attention_mask: mx.array | None,
        positions: mx.array | None,
    ) -> tuple[int, int]:
        if sequence.ndim != 3 or int(sequence.shape[-1]) != self.runner.dit.input_size:
            raise ValueError(
                "cached DiT sequence must have shape "
                f"(batch, sequence, {self.runner.dit.input_size})"
            )
        total_length = int(sequence.shape[1])
        fm_sequence_length = total_length - self.patch_size
        prefix_length = fm_sequence_length - self.hidden_patch_size
        if prefix_length < 0:
            raise ValueError("cached DiT sequence does not reserve a current unit")
        if prefix_length % self.unit_length:
            raise ValueError(
                "cached DiT finalized history must be unit-aligned: "
                f"prefix={prefix_length} unit={self.unit_length}"
            )
        patch_count = prefix_length // self.unit_length + 1
        if patch_count > state.max_patches:
            raise ValueError(
                "cached DiT request exceeds its capacity: "
                f"required_patches={patch_count} capacity={state.max_patches}"
            )
        if self.mode == "soar":
            if cfg_sequence is None or cfg_sequence.shape != sequence.shape:
                raise ValueError(
                    "cached SOAR conditional and CFG sequences must have equal shape"
                )
        elif cfg_sequence is not None:
            raise ValueError("cached MeanFlow does not accept a CFG sequence")
        if attention_mask is not None:
            if attention_mask.ndim not in {2, 3, 4} or attention_mask.shape[-2:] != (
                total_length,
                total_length,
            ):
                raise ValueError(
                    "cached DiT attention mask must cover the full input sequence"
                )
            if attention_mask.ndim in {3, 4} and attention_mask.shape[0] not in {
                1,
                int(sequence.shape[0]),
            }:
                raise ValueError(
                    "cached DiT attention mask batch must be one or match the input"
                )
        if positions is not None:
            if positions.ndim not in {1, 2} or int(positions.shape[-1]) != total_length:
                raise ValueError(
                    "cached DiT positions must match the full input sequence"
                )
            if positions.ndim == 2 and positions.shape[0] not in {
                1,
                int(sequence.shape[0]),
            }:
                raise ValueError(
                    "cached DiT position batch must be one or match the input"
                )
        return fm_sequence_length, prefix_length

    def sample(
        self,
        state: DiTSolverState,
        *,
        sequence: mx.array,
        cfg_sequence: mx.array | None = None,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        speaker_condition: mx.array | None = None,
        guidance_scale: float = 1.2,
        steps: int | None = None,
        noise: mx.array | None = None,
        _persistent_length: int | None = None,
    ) -> mx.array:
        """Generate one patch and commit only the preceding finalized unit."""

        compact_tail = _persistent_length is not None
        if compact_tail:
            if attention_mask is not None or positions is not None:
                raise ValueError("compact cached DiT tails derive mask and positions")
            if (
                _persistent_length < 0
                or _persistent_length % self.unit_length
            ):
                raise ValueError("compact cached DiT history must be unit-aligned")
            expected_length = 2 * self.unit_length
            if (
                sequence.ndim != 3
                or int(sequence.shape[1]) != expected_length
                or int(sequence.shape[-1]) != self.runner.dit.input_size
            ):
                raise ValueError(
                    "compact cached DiT sequence must contain previous and current units"
                )
            if self.mode == "soar":
                if cfg_sequence is None or cfg_sequence.shape != sequence.shape:
                    raise ValueError(
                        "compact cached SOAR tails require matching CFG state"
                    )
            elif cfg_sequence is not None:
                raise ValueError("compact cached MeanFlow tails do not accept CFG state")
            local_prefix_length = self.unit_length
            prefix_length = _persistent_length + self.unit_length
            patch_count = prefix_length // self.unit_length + 1
            if patch_count > state.max_patches:
                raise ValueError(
                    "cached DiT request exceeds its capacity: "
                    f"required_patches={patch_count} "
                    f"capacity={state.max_patches}"
                )
        else:
            _fm_sequence_length, prefix_length = self._validate_inputs(
                state,
                sequence=sequence,
                cfg_sequence=cfg_sequence,
                attention_mask=attention_mask,
                positions=positions,
            )
            local_prefix_length = prefix_length
        batch_size = int(sequence.shape[0])
        expected_noise = (batch_size, self.patch_size, self.latent_dim)
        coordinate = (
            mx.random.normal(expected_noise).astype(sequence.dtype)
            if noise is None
            else noise.astype(sequence.dtype)
        )
        if coordinate.shape != expected_noise:
            raise ValueError(f"cached DiT noise must have shape {expected_noise}")
        schedule, modulations_by_nfe = self._prepare_request(
            state,
            speaker_condition=speaker_condition,
            steps=steps,
            batch_size=batch_size,
            dtype=sequence.dtype,
        )
        nfe = int(schedule.times.shape[0])
        self._validate_state_cache(state, nfe=nfe, batch_size=batch_size)
        current_start = local_prefix_length
        current_end = current_start + self.hidden_patch_size
        current_hidden = sequence[:, current_start:current_end]
        cfg_current_hidden = (
            None if cfg_sequence is None else cfg_sequence[:, current_start:current_end]
        )
        total_length = prefix_length + self.unit_length
        if compact_tail:
            full_positions = mx.arange(
                prefix_length - self.unit_length,
                total_length,
                dtype=mx.float32,
            )[None]
        else:
            full_positions = (
                mx.arange(total_length, dtype=mx.float32)[None]
                if positions is None
                else positions
            )
        full_positions = _expand_branch_positions(
            full_positions,
            batch_size=batch_size,
            branch_count=self.runner.branch_count,
        )
        branch_attention_mask = _expand_branch_mask(
            attention_mask,
            batch_size=batch_size,
            branch_count=self.runner.branch_count,
        )
        if prefix_length == 0:
            if state.cache is not None and state.cache.valid_tokens != 0:
                raise ValueError("DiT solver state contains history for a first patch")
            first_mask = branch_attention_mask
            first_positions = full_positions[..., : self.unit_length]
            for nfe_index, modulations in enumerate(modulations_by_nfe):
                velocity = self.runner.first_velocity(
                    coordinate,
                    current_hidden=current_hidden,
                    cfg_current_hidden=cfg_current_hidden,
                    modulations=modulations,
                    positions=first_positions,
                    attention_mask=first_mask,
                    guidance_scale=guidance_scale,
                )
                coordinate = coordinate + schedule.step_size * velocity
            mx.eval(coordinate)
            return coordinate

        persistent_length = prefix_length - self.unit_length
        previous_unit = sequence[:, : self.unit_length] if compact_tail else sequence[
            :, persistent_length:prefix_length
        ]
        cfg_previous_unit = (
            None
            if cfg_sequence is None
            else (
                cfg_sequence[:, : self.unit_length]
                if compact_tail
                else cfg_sequence[:, persistent_length:prefix_length]
            )
        )
        cache = state.cache
        if compact_tail and persistent_length and cache is None:
            raise ValueError("compact cached DiT history requires an existing cache")
        if cache is None and persistent_length:
            required_prefix_mask = _causal_mask(persistent_length)
            prefix_mask = (
                required_prefix_mask
                if branch_attention_mask is None
                else _enforce_required_mask(
                    _mask_slice(
                        branch_attention_mask,
                        query_start=0,
                        query_end=persistent_length,
                        key_end=persistent_length,
                    ),
                    required_prefix_mask,
                )
            )
            prefill_writes: list[tuple[int, mx.array, mx.array]] = []
            for nfe_index, modulations in enumerate(modulations_by_nfe):
                keys, values = self.runner.prefill_nfe(
                    prefix_sequence=sequence[:, :persistent_length],
                    cfg_prefix_sequence=(
                        None
                        if cfg_sequence is None
                        else cfg_sequence[:, :persistent_length]
                    ),
                    modulations=modulations,
                    positions=full_positions[..., :persistent_length],
                    attention_mask=prefix_mask,
                )
                prefill_writes.append((nfe_index, keys, values))
            _first_nfe, first_keys, first_values = prefill_writes[0]
            cache = self._allocate_projected_cache(
                state,
                nfe=nfe,
                batch_size=batch_size,
                keys=first_keys,
                values=first_values,
            )
            cache = self._grow_projected_cache(
                cache,
                required_tokens=persistent_length,
            )
            for nfe_index, keys, values in prefill_writes:
                cache.validate_write(nfe_index, keys, values)
            for nfe_index, keys, values in prefill_writes:
                cache.write(nfe_index, keys, values)
            mx.eval(cache.cache_k, cache.cache_v)
        if cache is not None and any(
            offset != persistent_length for offset in cache.offsets
        ):
            raise ValueError(
                "DiT solver state offset does not match finalized history: "
                f"offsets={cache.offsets} expected={persistent_length}"
            )
        if cache is not None:
            cache = self._grow_projected_cache(
                cache,
                required_tokens=persistent_length + 2 * self.unit_length,
            )
        required_tail_mask = _delayed_tail_mask(persistent_length, self.unit_length)
        tail_mask = (
            required_tail_mask
            if branch_attention_mask is None
            else _enforce_required_mask(
                _mask_slice(
                    branch_attention_mask,
                    query_start=persistent_length,
                    query_end=total_length,
                    key_end=total_length,
                ),
                required_tail_mask,
            )
        )
        tail_positions = (
            full_positions
            if compact_tail
            else full_positions[..., persistent_length:total_length]
        )
        cache_factory: CacheFactory | None = None
        if cache is None:

            def allocate_scratch_cache(
                key_dtype: mx.Dtype,
                value_dtype: mx.Dtype,
            ) -> DiTKvCache:
                return self.runner.allocate_cache(
                    capacity_patches=DIT_CACHE_BUCKETS[0],
                    nfe=nfe,
                    batch_size=batch_size,
                    key_dtype=key_dtype,
                    value_dtype=value_dtype,
                )

            cache_factory = allocate_scratch_cache

        for nfe_index, modulations in enumerate(modulations_by_nfe):
            velocity, cache = self.runner.next_velocity(
                coordinate,
                previous_unit=previous_unit,
                current_hidden=current_hidden,
                cfg_previous_unit=cfg_previous_unit,
                cfg_current_hidden=cfg_current_hidden,
                cache=cache,
                cache_factory=cache_factory,
                nfe_index=nfe_index,
                modulations=modulations,
                positions=tail_positions,
                attention_mask=tail_mask,
                guidance_scale=guidance_scale,
            )
            coordinate = coordinate + schedule.step_size * velocity
        if cache is None:
            raise RuntimeError("cached DiT did not produce request K/V")
        mx.eval(coordinate, cache.cache_k, cache.cache_v)
        cache.publish_unit()
        if state.cache is not cache:
            state.cache = cache
        return coordinate

    def sample_tail(
        self,
        state: DiTSolverState,
        *,
        previous_unit: mx.array,
        current_hidden: mx.array,
        cfg_previous_unit: mx.array | None = None,
        cfg_current_hidden: mx.array | None = None,
        speaker_condition: mx.array | None = None,
        guidance_scale: float = 1.2,
        steps: int | None = None,
        noise: mx.array | None = None,
    ) -> mx.array:
        """Generate from the fixed fresh tail while persistent history stays cached."""

        batch_size = int(previous_unit.shape[0])
        expected_previous = (
            batch_size,
            self.unit_length,
            self.runner.dit.input_size,
        )
        expected_hidden = (
            batch_size,
            self.hidden_patch_size,
            self.runner.dit.input_size,
        )
        if previous_unit.shape != expected_previous:
            raise ValueError(
                f"cached DiT previous unit must have shape {expected_previous}"
            )
        if current_hidden.shape != expected_hidden:
            raise ValueError(
                f"cached DiT current hidden must have shape {expected_hidden}"
            )
        padding = mx.zeros(
            (batch_size, self.patch_size, self.runner.dit.input_size),
            dtype=current_hidden.dtype,
        )
        sequence = mx.concatenate((previous_unit, current_hidden, padding), axis=1)
        cfg_sequence = None
        if self.mode == "soar":
            if cfg_previous_unit is None or cfg_current_hidden is None:
                raise ValueError("cached SOAR tail requires conditional and CFG units")
            if (
                cfg_previous_unit.shape != expected_previous
                or cfg_current_hidden.shape != expected_hidden
            ):
                raise ValueError("cached SOAR tail CFG shapes differ from conditional")
            cfg_sequence = mx.concatenate(
                (cfg_previous_unit, cfg_current_hidden, padding), axis=1
            )
        elif cfg_previous_unit is not None or cfg_current_hidden is not None:
            raise ValueError("cached MeanFlow tail does not accept CFG units")
        persistent_length = 0 if state.cache is None else state.cache.valid_tokens
        return self.sample(
            state,
            sequence=sequence,
            cfg_sequence=cfg_sequence,
            speaker_condition=speaker_condition,
            guidance_scale=guidance_scale,
            steps=steps,
            noise=noise,
            _persistent_length=persistent_length,
        )


class CachedMeanFlowSolver(CachedDiTSolver):
    def __init__(
        self,
        dit: DiT,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
        patch_size: int = 4,
        hidden_patch_size: int = 1,
    ):
        super().__init__(
            dit,
            coordinate_projection,
            latent_dim=latent_dim,
            patch_size=patch_size,
            hidden_patch_size=hidden_patch_size,
            mode="meanflow",
        )


class CachedSOARSolver(CachedDiTSolver):
    def __init__(
        self,
        dit: DiT,
        coordinate_projection: nn.Module,
        *,
        latent_dim: int,
        patch_size: int = 4,
        hidden_patch_size: int = 1,
    ):
        super().__init__(
            dit,
            coordinate_projection,
            latent_dim=latent_dim,
            patch_size=patch_size,
            hidden_patch_size=hidden_patch_size,
            mode="soar",
        )


__all__ = [
    "DIT_CACHE_BUCKETS",
    "CachedDiTRunner",
    "CachedDiTSolver",
    "CachedMeanFlowSolver",
    "CachedSOARSolver",
    "DiTKvCache",
    "DiTSolverState",
    "resolve_dit_cache_bucket",
]
