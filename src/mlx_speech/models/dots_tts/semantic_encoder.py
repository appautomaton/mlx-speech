"""Causal MLX semantic patch encoder for dots.tts autoregression."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import DotsTTSConfig
from .layers import CausalConv1d, SemanticEncoderLayer, SemanticLayerCache


@dataclass(frozen=True)
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

    def _project(self, value: mx.array) -> mx.array:
        batch, tokens, hidden = value.shape
        rate = self.output_downsample_rate
        if int(tokens) % rate:
            raise ValueError("semantic token count is not output-group aligned")
        grouped = value.reshape(batch, tokens // rate, rate * hidden)
        return self.out_proj(grouped)

    def _state_for(self, value: mx.array) -> SemanticEncoderState:
        empty = mx.zeros(
            (int(value.shape[0]), 0, self.encoder.num_heads, self.encoder.head_dim),
            dtype=value.dtype,
        )
        return SemanticEncoderState(
            conv_tail=mx.zeros(
                (int(value.shape[0]), self.ds_proj.left_padding, self.input_dim),
                dtype=value.dtype,
            ),
            layer_caches=tuple(
                SemanticLayerCache(keys=empty, values=empty)
                for _ in self.encoder.layers
            ),
        )

    def _increment(
        self, value: mx.array, state: SemanticEncoderState
    ) -> tuple[mx.array, SemanticEncoderState]:
        if int(state.conv_tail.shape[0]) != int(value.shape[0]):
            raise ValueError("semantic state batch size differs from input")
        if len(state.layer_caches) != len(self.encoder.layers):
            raise ValueError("semantic state layer count differs from encoder")
        conv_input = mx.concatenate((state.conv_tail, value), axis=1)
        downsampled = self.ds_proj._convolve(conv_input)
        next_tail = value[:, -self.ds_proj.left_padding :, :]
        encoded, caches = self.encoder(
            self.in_proj(downsampled), caches=state.layer_caches
        )
        next_length = state.sequence_length + int(encoded.shape[1])
        return self._project(encoded), SemanticEncoderState(
            conv_tail=next_tail,
            layer_caches=caches,
            sequence_length=next_length,
        )

    def prefill(
        self,
        value: mx.array,
        state: SemanticEncoderState | None = None,
    ) -> tuple[mx.array, SemanticEncoderState]:
        self._validate_input(value)
        current = self._state_for(value) if state is None else state
        return self._increment(value, current)

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
    ) -> tuple[mx.array, tuple[SemanticLayerCache, ...]]:
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError("semantic cache layer count differs from model")
        next_caches = []
        for index, layer in enumerate(self.layers):
            cache = None if caches is None else caches[index]
            value, next_cache = layer(value, cache=cache)
            next_caches.append(next_cache)
        return value, tuple(next_caches)


__all__ = ["SemanticEncoderState", "VAESemanticEncoder"]
