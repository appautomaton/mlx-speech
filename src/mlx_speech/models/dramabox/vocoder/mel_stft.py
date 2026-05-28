"""STFT-as-conv1d helpers for the BWE's mel front-end.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/vocoder.py:419-494`

The upstream `_STFTFn` precomputes ``forward_basis`` and ``inverse_basis``
buffers (shape ``[2 * n_freqs, 1, filter_length]``) and uses them as the
weight for a Conv1d / ConvTranspose1d that performs windowed DFT. The first
``n_freqs`` rows are real components, the next ``n_freqs`` are imaginary.

The mel-filterbank `mel_basis` is also stored in the checkpoint.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class _STFTFn(nn.Module):
    """STFT via depthwise-grouped conv with the precomputed DFT × Hann basis."""

    def __init__(self, *, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        # Buffer shapes per checkpoint: [2 * n_freqs, 1, filter_length]
        self.forward_basis = mx.zeros((2 * n_freqs, 1, filter_length), dtype=mx.float32)
        self.inverse_basis = mx.zeros((2 * n_freqs, 1, filter_length), dtype=mx.float32)

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        """Compute (magnitude, phase) from a batch of waveforms ``(B, T)``.

        Output shape:
            magnitude: ``(B, n_freqs, T_frames)``
            phase:     ``(B, n_freqs, T_frames)``
        """
        if y.ndim == 2:
            y = y[:, None, :]  # (B, 1, T)
        left_pad = max(0, self.win_length - self.hop_length)
        # Pad on the LEFT only (causal)
        pad_l = mx.zeros((y.shape[0], y.shape[1], left_pad), dtype=y.dtype)
        y = mx.concatenate([pad_l, y], axis=-1)
        # Run a Conv1d with the saved forward_basis as the weight.
        # MLX wants channel-last: (B, T, C_in=1). forward_basis is
        # (out=2*n_freqs, 1, K=filter_length) in PyTorch layout. MLX Conv1d
        # weight is (out, K, in), which matches directly.
        y_cl = y.transpose(0, 2, 1)  # (B, T, 1)
        w = self.forward_basis  # (2*n_freqs, 1, filter_length) → (out, K=in_K, in_C=1)?
        # Actually the saved basis is shape (2*n_freqs, 1, filter_length) = (out, in, K) PyTorch.
        # MLX wants (out, K, in) → permute (0, 2, 1).
        w = w.transpose(0, 2, 1)  # (out, K, in=1)
        spec = mx.conv1d(y_cl, w.astype(y_cl.dtype), stride=self.hop_length, padding=0)
        # spec shape: (B, T_frames, 2*n_freqs) → (B, 2*n_freqs, T_frames)
        spec = spec.transpose(0, 2, 1)
        n_freqs = spec.shape[1] // 2
        real = spec[:, :n_freqs]
        imag = spec[:, n_freqs:]
        magnitude = mx.sqrt(real * real + imag * imag)
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel front-end for the BWE's input.

    Saved keys:
        stft_fn.forward_basis     [2*n_freqs, 1, filter_length]
        stft_fn.inverse_basis     [2*n_freqs, 1, filter_length]
        mel_basis                 [n_mel_channels, n_freqs]
    """

    def __init__(
        self,
        *,
        filter_length: int = 512,
        hop_length: int = 80,
        win_length: int = 512,
        n_mel_channels: int = 64,
    ):
        super().__init__()
        self.stft_fn = _STFTFn(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
        )
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs), dtype=mx.float32)

    def mel_spectrogram(self, y: mx.array) -> mx.array:
        """Log-mel from waveform ``(B, T)``. Returns ``(B, n_mel, T_frames)``."""
        magnitude, _ = self.stft_fn(y)  # (B, n_freqs, T_frames)
        mel = mx.matmul(self.mel_basis.astype(magnitude.dtype), magnitude)
        log_mel = mx.log(mx.maximum(mel, mx.array(1e-5, dtype=mel.dtype)))
        return log_mel


__all__ = ["MelSTFT"]
