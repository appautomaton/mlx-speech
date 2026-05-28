"""Anti-aliased activation: upsample → snake → downsample.

The activation wrapper takes a channel-first input ``(B, C, T)``, upsamples
in time, applies the snake activation, then downsamples back. Both the
upsample and downsample low-pass filters are pre-computed (kaiser-sinc) and
stored in the checkpoint as buffers — we do NOT regenerate them.

Saved sub-keys (per activation):
    act.alpha           [C]   (snake_beta per-channel alpha, log-scale)
    act.beta            [C]   (snake_beta per-channel beta,  log-scale)
    upsample.filter            [1, 1, kernel_size]   pre-computed
    downsample.lowpass.filter  [1, 1, kernel_size]   pre-computed

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/vocoder.py:51-162`
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .snake import SnakeBeta


# --------------------------------------------------------------------------- #
# LowPass / UpSample / DownSample
# --------------------------------------------------------------------------- #

class _LowPassFilter1d(nn.Module):
    """Apply a depthwise-grouped 1D conv with a fixed low-pass filter.

    The filter is loaded as ``self.filter`` of shape ``(1, 1, kernel_size)``;
    we depthwise-broadcast across channels at apply time.
    """

    def __init__(self, *, kernel_size: int = 12, stride: int = 1):
        super().__init__()
        # Stored as the saved checkpoint buffer: [1, 1, kernel_size]
        self.filter = mx.zeros((1, 1, kernel_size), dtype=mx.float32)
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        """Input shape (B, C, T) — apply depthwise conv with shared filter."""
        # Replicate-pad the time dimension
        x_padded = _replicate_pad_1d(x, self.pad_left, self.pad_right)
        # Convolve depthwise: expand the filter to (C, kernel, 1) shape then
        # do a grouped conv with groups=C. MLX conv has no grouped flag, so
        # we implement depthwise via the trick: reshape (B, C, T) → (B*C, 1, T),
        # apply the (1, K, 1) filter, then reshape back.
        B, C, T = x_padded.shape
        # MLX Conv1d wants channel-last: (B, T, C). For grouped depthwise via
        # reshape trick, the cleanest path is per-channel sliding-window
        # multiply-and-sum. Avoid MLX's missing grouped conv by doing a
        # broadcast multiply + sum manually.
        return _depthwise_conv1d_with_filter(x_padded, self.filter, stride=self.stride)


class _UpSample1d(nn.Module):
    """Upsample-by-``ratio`` via stride-`ratio` ConvTranspose1d with a fixed
    low-pass filter.

    Implementation note: rather than rely on MLX's ConvTranspose1d with
    groups (which doesn't exist), we replicate the BigVGAN behavior via a
    nearest-neighbor-replicate upsample followed by a depthwise low-pass
    conv. The filter is the one saved in the checkpoint.
    """

    def __init__(self, *, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        # Stored as the saved checkpoint buffer: [1, 1, kernel_size]
        self.filter = mx.zeros((1, 1, kernel_size), dtype=mx.float32)
        self.ratio = ratio
        self.kernel_size = kernel_size
        # padding values copied from upstream UpSample1d kaiser-window branch
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (self.kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (self.kernel_size - ratio + 1) // 2

    def __call__(self, x: mx.array) -> mx.array:
        """Input ``(B, C, T)`` → ``(B, C, T * ratio)``.

        Mirrors upstream: replicate-pad by `self.pad`, then grouped
        `conv_transpose1d` with stride `ratio` and the saved filter, then
        slice off `pad_left` / `pad_right`. The factor `ratio` is multiplied
        in to keep the sinc-filter magnitude correct.
        """
        B, C, T = x.shape
        # Replicate-pad by self.pad on each side
        x = _replicate_pad_1d(x, self.pad, self.pad)
        # Grouped conv-transpose with the depthwise filter
        x_cl = x.transpose(0, 2, 1)  # (B, T, C)
        # filter is (1, 1, K) — expand to (C, K, 1) for groups=C
        w = mx.broadcast_to(self.filter.reshape(1, self.kernel_size, 1), (C, self.kernel_size, 1)).astype(x_cl.dtype)
        out = mx.conv_transpose1d(x_cl, w * float(self.ratio), stride=self.ratio, padding=0, groups=C)
        out = out.transpose(0, 2, 1)  # back to (B, C, T_out)
        # Slice off the kaiser pad
        return out[..., self.pad_left : out.shape[-1] - self.pad_right]


class _DownSample1d(nn.Module):
    """Stride-`ratio` low-pass conv that decimates by ``ratio``.

    The saved child is named ``lowpass`` (`LowPassFilter1d`) so we honor that
    nested key path.
    """

    def __init__(self, *, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = _LowPassFilter1d(kernel_size=kernel_size, stride=ratio)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


# --------------------------------------------------------------------------- #
# The anti-aliased Activation1d
# --------------------------------------------------------------------------- #

class Activation1d(nn.Module):
    """Up-sample 2× → SnakeBeta → down-sample 2×.

    Saved keys (per instance):
        act.alpha, act.beta             — SnakeBeta params
        upsample.filter                 — pre-computed low-pass filter
        downsample.lowpass.filter       — pre-computed low-pass filter
    """

    def __init__(self, channels: int, *, up_ratio: int = 2, down_ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = _UpSample1d(ratio=up_ratio, kernel_size=kernel_size)
        self.downsample = _DownSample1d(ratio=down_ratio, kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


# --------------------------------------------------------------------------- #
# Helpers: depthwise conv with a single shared filter (MLX has no `groups=`)
# --------------------------------------------------------------------------- #

def _replicate_pad_1d(x: mx.array, left: int, right: int) -> mx.array:
    """Replicate-pad the last axis (time) by `left` left and `right` right."""
    if left == 0 and right == 0:
        return x
    # x shape (B, C, T)
    pad_l = mx.broadcast_to(x[..., :1], (x.shape[0], x.shape[1], left)) if left > 0 else None
    pad_r = mx.broadcast_to(x[..., -1:], (x.shape[0], x.shape[1], right)) if right > 0 else None
    parts: list[mx.array] = []
    if pad_l is not None:
        parts.append(pad_l)
    parts.append(x)
    if pad_r is not None:
        parts.append(pad_r)
    return mx.concatenate(parts, axis=-1)


def _depthwise_conv1d_with_filter(x: mx.array, filt: mx.array, *, stride: int) -> mx.array:
    """Apply the same 1D filter to every channel of (B, C, T_in).

    ``filt`` has shape ``(1, 1, kernel_size)``. We broadcast to ``(C, kernel, 1)``
    and call `mx.conv1d` with ``groups=C``. MLX expects channel-last input
    so we transpose at the boundary.
    """
    B, C, T_in = x.shape
    K = filt.shape[-1]
    # MLX wants (B, T, C) input
    x_cl = x.transpose(0, 2, 1)  # (B, T, C)
    # Broadcast the filter to (C, K, 1) — every channel gets the same kernel
    w = mx.broadcast_to(filt.reshape(1, K, 1), (C, K, 1)).astype(x.dtype)
    out = mx.conv1d(x_cl, w, stride=stride, padding=0, groups=C)
    # (B, T_out, C) → (B, C, T_out)
    return out.transpose(0, 2, 1)


def _upsample_with_zeros(x: mx.array, ratio: int) -> mx.array:
    """Insert `ratio - 1` zeros between consecutive time-samples per channel.

    Equivalent to a ConvTranspose1d with `stride=ratio` and an identity
    kernel.
    """
    if ratio == 1:
        return x
    B, C, T = x.shape
    # Build (B, C, T, ratio) where index 0 has the original sample and the rest are zero
    expanded = mx.zeros((B, C, T, ratio), dtype=x.dtype)
    expanded = mx.concatenate([x[..., None], mx.zeros((B, C, T, ratio - 1), dtype=x.dtype)], axis=-1)
    # Reshape to (B, C, T * ratio)
    return expanded.reshape(B, C, T * ratio)


class HannSincUpsampler:
    """Hann-windowed sinc upsampler — port of the ``window_type="hann"`` branch
    of the reference ``UpSample1d`` (`vocoder.py:82-127`), which matches
    ``torchaudio.functional.resample``.

    Used for the BWE skip connection (16 kHz → 48 kHz). The filter is built at
    construction (the reference marks it ``persistent=False`` — it is NOT a
    checkpoint key), so this is a plain object rather than an ``nn.Module`` to
    keep the filter out of the parameter tree (otherwise ``load_weights(strict)``
    would demand a checkpoint value for it).
    """

    def __init__(self, ratio: int):
        self.ratio = ratio
        rolloff = 0.99
        lowpass_filter_width = 6
        width = math.ceil(lowpass_filter_width / rolloff)
        self.kernel_size = 2 * width * ratio + 1
        self.pad = width
        self.pad_left = 2 * width * ratio
        self.pad_right = self.kernel_size - ratio

        # Build the filter in float64 numpy, store float32 (RoPE-style boundary).
        t = (np.arange(self.kernel_size) / ratio - width) * rolloff
        t_clamped = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
        window = np.cos(t_clamped * np.pi / lowpass_filter_width / 2) ** 2
        filt = (np.sinc(t) * window * rolloff / ratio).astype(np.float32)
        self._filter = mx.array(filt.reshape(1, 1, self.kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        """``(B, C, T)`` → ``(B, C, T * ratio)`` via depthwise transposed conv."""
        B, C, _ = x.shape
        x = _replicate_pad_1d(x, self.pad, self.pad)
        x_cl = x.transpose(0, 2, 1)  # (B, T, C)
        w = mx.broadcast_to(
            self._filter.reshape(1, self.kernel_size, 1), (C, self.kernel_size, 1)
        ).astype(x_cl.dtype)
        out = mx.conv_transpose1d(x_cl, w * float(self.ratio), stride=self.ratio, padding=0, groups=C)
        out = out.transpose(0, 2, 1)  # (B, C, T_out)
        return out[..., self.pad_left : out.shape[-1] - self.pad_right]


__all__ = [
    "Activation1d",
    "HannSincUpsampler",
]
