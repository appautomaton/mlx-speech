"""Causal MLX semantic patch encoder for dots.tts autoregression."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import DotsTTSConfig
from .layers import CausalConv1d, SemanticEncoderLayer, SemanticLayerCache


_DEFAULT_SEMANTIC_CACHE_PATCHES = 512


@dataclass
class SemanticEncoderState:
    conv_tail: mx.array
    layer_caches: tuple[SemanticLayerCache, ...]
    sequence_length: int = 0


class VAESemanticEncoder(nn.Module):
    """Map continuous VAE latent patches into Qwen hidden embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        patch_size: int = 4,
    ):
        super().__init__()
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("semantic patch_size must be a positive multiple of 2")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)
        self.patch_size = int(patch_size)
        self.input_downsample_rate = 2
        self.output_downsample_rate = self.patch_size // self.input_downsample_rate
        self.expects_normalized_input = False
        self.ds_proj = CausalConv1d(
            input_dim,
            input_dim,
            kernel_size=self.input_downsample_rate,
            stride=self.input_downsample_rate,
        )
        self.in_proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.encoder = _SemanticTransformer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
        )
        self.out_proj = nn.Linear(
            hidden_size * self.output_downsample_rate,
            output_dim,
            bias=True,
        )

    @classmethod
    def from_config(
        cls, config: DotsTTSConfig, *, output_dim: int
    ) -> "VAESemanticEncoder":
        encoder = config.patch_encoder
        return cls(
            input_dim=config.latent_dim,
            hidden_size=encoder.hidden_size,
            output_dim=output_dim,
            num_layers=encoder.num_layers,
            num_heads=encoder.num_heads,
            intermediate_size=encoder.ffn_hidden_size,
            patch_size=config.patch_size,
        )

    def _validate_input(self, value: mx.array, *, exact_patch: bool = False) -> None:
        if value.ndim != 3 or int(value.shape[-1]) != self.input_dim:
            raise ValueError(
                f"semantic input must have shape (batch, time, {self.input_dim}), "
                f"got {value.shape}"
            )
        time = int(value.shape[1])
        if exact_patch and time != self.patch_size:
            raise ValueError(
                f"semantic decode patch must have length {self.patch_size}, got {time}"
            )
        if not exact_patch and (time <= 0 or time % self.patch_size):
            raise ValueError(
                f"semantic input time must be divisible by {self.patch_size}, got {time}"
            )

    def fuse_for_inference(self) -> None:
        """Fuse self-attention projections after strict checkpoint loading."""

        for layer in self.encoder.layers:
            layer.attn.fuse_qkv_for_inference()

    def _project(self, value: mx.array) -> mx.array:
        batch, tokens, hidden = value.shape
        rate = self.output_downsample_rate
        if int(tokens) % rate:
            raise ValueError("semantic token count is not output-group aligned")
        grouped = value.reshape(batch, tokens // rate, rate * hidden)
        return self.out_proj(grouped)

    def _increment(
        self,
        value: mx.array,
        state: SemanticEncoderState | None,
        *,
        cache_capacity: int | None = None,
    ) -> tuple[mx.array, SemanticEncoderState]:
        if state is None:
            conv_tail = mx.zeros(
                (int(value.shape[0]), self.ds_proj.left_padding, self.input_dim),
                dtype=value.dtype,
            )
            layer_caches = None
            sequence_length = 0
        else:
            expected_tail = (
                int(value.shape[0]),
                self.ds_proj.left_padding,
                self.input_dim,
            )
            if state.conv_tail.shape != expected_tail:
                raise ValueError(
                    "semantic state batch size or convolution tail is invalid"
                )
            if len(state.layer_caches) != len(self.encoder.layers):
                raise ValueError("semantic state layer count differs from encoder")
            if any(
                cache.offset != state.sequence_length for cache in state.layer_caches
            ):
                raise ValueError(
                    "semantic state cache offsets differ from sequence_length"
                )
            if cache_capacity is not None and any(
                cache.capacity != cache_capacity for cache in state.layer_caches
            ):
                raise ValueError(
                    "semantic state cache capacity differs from the request capacity"
                )
            conv_tail = state.conv_tail
            layer_caches = state.layer_caches
            sequence_length = state.sequence_length

        prior_offsets = (
            tuple(cache.offset for cache in layer_caches)
            if layer_caches is not None
            else ()
        )
        try:
            conv_input = mx.concatenate((conv_tail, value), axis=1)
            downsampled = self.ds_proj._convolve(conv_input)
            next_tail = value[:, -self.ds_proj.left_padding :, :]
            encoded, caches = self.encoder(
                self.in_proj(downsampled),
                caches=layer_caches,
                cache_capacity=cache_capacity,
            )
            next_length = sequence_length + int(encoded.shape[1])
            projected = self._project(encoded)
        except Exception:
            if layer_caches is not None:
                for cache, prior_offset in zip(
                    layer_caches, prior_offsets, strict=True
                ):
                    cache.restore_offset(prior_offset)
            raise

        if state is None:
            state = SemanticEncoderState(
                conv_tail=next_tail,
                layer_caches=caches,
                sequence_length=next_length,
            )
        else:
            state.conv_tail = next_tail
            state.layer_caches = caches
            state.sequence_length = next_length
        return projected, state

    def prefill(
        self,
        value: mx.array,
        state: SemanticEncoderState | None = None,
        *,
        max_audio_patches: int | None = None,
    ) -> tuple[mx.array, SemanticEncoderState]:
        self._validate_input(value)
        cache_capacity = None
        if state is None or max_audio_patches is not None:
            patch_capacity = (
                _DEFAULT_SEMANTIC_CACHE_PATCHES
                if max_audio_patches is None
                else int(max_audio_patches)
            )
            if patch_capacity <= 0:
                raise ValueError("semantic max_audio_patches must be positive")
            cache_capacity = patch_capacity * self.output_downsample_rate
        return self._increment(value, state, cache_capacity=cache_capacity)

    def decode_patch(
        self,
        patch: mx.array,
        state: SemanticEncoderState,
    ) -> tuple[mx.array, SemanticEncoderState]:
        self._validate_input(patch, exact_patch=True)
        return self._increment(patch, state)

    def __call__(self, value: mx.array) -> mx.array:
        self._validate_input(value)
        downsampled = self.ds_proj(value)
        encoded, _ = self.encoder(self.in_proj(downsampled))
        return self._project(encoded)


class _SemanticTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("semantic encoder must have at least one layer")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // int(num_heads)
        self.layers = [
            SemanticEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        value: mx.array,
        *,
        caches: tuple[SemanticLayerCache, ...] | None = None,
        cache_capacity: int | None = None,
    ) -> tuple[mx.array, tuple[SemanticLayerCache, ...]]:
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError("semantic cache layer count differs from model")
        prior_offsets = (
            tuple(cache.offset for cache in caches) if caches is not None else ()
        )
        if caches is not None:
            for cache in caches:
                cache.validate_append_length(int(value.shape[1]))
        next_caches = []
        try:
            for index, layer in enumerate(self.layers):
                cache = None if caches is None else caches[index]
                value, next_cache = layer(
                    value,
                    cache=cache,
                    cache_capacity=cache_capacity,
                )
                next_caches.append(next_cache)
        except Exception:
            if caches is not None:
                for cache, prior_offset in zip(caches, prior_offsets, strict=True):
                    cache.restore_offset(prior_offset)
            raise
        return value, tuple(next_caches)


__all__ = ["SemanticEncoderState", "VAESemanticEncoder"]
