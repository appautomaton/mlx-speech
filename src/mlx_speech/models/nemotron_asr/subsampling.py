"""Causal depthwise-striding subsampling for Nemotron 3.5 ASR."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def subsampled_length(length: int, stages: int = 3) -> int:
    """Apply NeMo's causal ``k=3, stride=2, padding=(2, 1)`` recurrence."""
    if length < 0:
        raise ValueError("length must be non-negative")
    for _ in range(stages):
        length = (length + 3 - 3) // 2 + 1
    return length


class CausalDwStridingSubsampling(nn.Module):
    """Three causal stride-2 stages reducing mel frames by a factor of eight.

    The module names and list indices match NeMo's ``encoder.pre_encode``
    checkpoint paths: stride convolutions live at ``conv.0``, ``conv.2``, and
    ``conv.5``; pointwise convolutions live at ``conv.3`` and ``conv.6``.
    """

    def __init__(
        self,
        *,
        feat_in: int = 128,
        d_model: int = 1024,
        conv_channels: int = 256,
        subsampling_factor: int = 8,
    ) -> None:
        super().__init__()
        stages = int(math.log2(subsampling_factor))
        if 2**stages != subsampling_factor:
            raise ValueError("subsampling_factor must be a power of two")
        if stages < 1:
            raise ValueError("subsampling_factor must be at least two")

        self.feat_in = feat_in
        self.d_model = d_model
        self.conv_channels = conv_channels
        self.subsampling_factor = subsampling_factor
        self.stages = stages
        self.kernel_size = 3
        self.stride = 2
        self.pad_left = self.kernel_size - 1
        self.pad_right = self.stride - 1

        layers: list[nn.Module] = [
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        ]
        for _ in range(stages - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        groups=conv_channels,
                    ),
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=1),
                    nn.ReLU(),
                ]
            )
        self.conv = layers
        self._strided_indices = frozenset({0} | {2 + 3 * i for i in range(stages - 1)})

        frequency = subsampled_length(feat_in, stages)
        self.out = nn.Linear(conv_channels * frequency, d_model)

    def output_lengths(self, lengths: mx.array) -> mx.array:
        values = lengths.astype(mx.int32)
        for _ in range(self.stages):
            values = values // self.stride + 1
        return values

    @staticmethod
    def _mask_time(x: mx.array, lengths: mx.array) -> mx.array:
        valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
        return x * valid[:, :, None, None].astype(x.dtype)

    def __call__(self, features: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        if features.ndim != 3:
            raise ValueError(f"expected features [B, T, F], got {features.shape}")
        if features.shape[0] != lengths.shape[0]:
            raise ValueError("batch dimension and lengths must agree")
        if features.shape[2] != self.feat_in:
            raise ValueError(
                f"expected {self.feat_in} mel bins, got {features.shape[2]}"
            )

        x = features[:, :, :, None]
        current_lengths = lengths.astype(mx.int32)
        for index, layer in enumerate(self.conv):
            if index in self._strided_indices:
                x = mx.pad(
                    x,
                    (
                        (0, 0),
                        (self.pad_left, self.pad_right),
                        (self.pad_left, self.pad_right),
                        (0, 0),
                    ),
                )
                current_lengths = current_lengths // self.stride + 1
            x = layer(x)
            if index in self._strided_indices:
                x = self._mask_time(x, current_lengths)

        batch, time, frequency, channels = x.shape
        # NeMo flattens [C, F] after transposing NCHW to [B, T, C, F].
        x = mx.transpose(x, (0, 1, 3, 2)).reshape(
            batch, time, channels * frequency
        )
        x = self.out(x)
        valid = mx.arange(time)[None, :] < current_lengths[:, None]
        x = x * valid[:, :, None].astype(x.dtype)
        return x, current_lengths


__all__ = ["CausalDwStridingSubsampling", "subsampled_length"]
