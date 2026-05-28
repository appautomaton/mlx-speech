"""VocoderWithBWE — main vocoder + BWE generator + sinc-resampled skip.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/vocoder.py:497-594`

Pipeline:

    mel [B, 2, T, 64]
        ──> main Vocoder ──> wav_16k [B, 2, T_16k]
                                   ├──> stereo-wise MelSTFT ──> BWE Vocoder ──> residual_48k
                                   └─────────────────────────> sinc-resample × 3 ──> skip_48k
                          ┌──────────────────┐
                          │  residual + skip │ → clamp to [-1, 1] → wav_48k
                          └──────────────────┘

The main vocoder runs in fp32; we upcast the input mel at the boundary and
let the inner ops promote the bf16 weights per-op.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .anti_aliased import HannSincUpsampler
from .mel_stft import MelSTFT
from .vocoder import Vocoder, VocoderArgs


class VocoderWithBWE(nn.Module):
    """Full vocoder + BWE chain.

    Args:
        main_args: config for the main vocoder (mel → 16 kHz wav).
        bwe_args: config for the BWE vocoder (16 kHz mel → 48 kHz residual).
        input_sampling_rate: 16_000.
        output_sampling_rate: 48_000.
        hop_length: BWE's STFT hop (80).
    """

    def __init__(
        self,
        *,
        main_args: VocoderArgs,
        bwe_args: VocoderArgs,
        input_sampling_rate: int = 16_000,
        output_sampling_rate: int = 48_000,
        hop_length: int = 80,
        n_fft: int = 512,
        win_length: int = 512,
        n_mel_channels: int = 64,
    ):
        super().__init__()
        self.vocoder = Vocoder(main_args)
        self.bwe_generator = Vocoder(bwe_args)
        self.mel_stft = MelSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
        )
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.ratio = output_sampling_rate // input_sampling_rate
        # Skip-connection resampler (16 kHz → 48 kHz). Hann-windowed sinc,
        # matching the reference (`vocoder.py:497-594` uses UpSample1d with
        # window_type="hann"). Plain object — its filter is built here, not
        # loaded from the checkpoint.
        self.skip_resampler = HannSincUpsampler(self.ratio)

    def __call__(self, mel_spec: mx.array) -> mx.array:
        """Forward.

        Args:
            mel_spec: ``(B, 2, T, mel_bins=64)`` stereo mel-spectrogram.

        Returns:
            ``(B, 2, T_out)`` waveform clipped to ``[-1, 1]`` at the output
            sample rate (48 kHz).
        """
        # Run main vocoder in fp32 (bf16 inputs upcast; weights upcast per-op
        # via inner operations).
        input_dtype = mel_spec.dtype
        mel32 = mel_spec.astype(mx.float32)
        x = self.vocoder(mel32)  # (B, 2, T_16k) at 16 kHz
        T_low = x.shape[-1]
        output_length = T_low * self.output_sampling_rate // self.input_sampling_rate

        # Pad to a multiple of hop_length for exact mel frame count
        remainder = T_low % self.hop_length
        if remainder != 0:
            pad = mx.zeros((x.shape[0], x.shape[1], self.hop_length - remainder), dtype=x.dtype)
            x = mx.concatenate([x, pad], axis=-1)

        # Compute mel from the 16 kHz wav: (B, 2, n_mel, T_frames)
        mel = self._compute_mel(x)
        # BWE expects (B, 2, T, mel_bins) — transpose to time-then-mel
        mel_for_bwe = mel.transpose(0, 1, 3, 2)
        residual = self.bwe_generator(mel_for_bwe)  # (B, 2, T_48k)

        # Hann-windowed sinc resampler for the skip path (16 kHz → 48 kHz).
        # Zero-order hold (mx.repeat) here injects imaging artifacts that read
        # as a synthetic/metallic coloration, since the skip is summed directly
        # into the output. Match the reference's windowed-sinc resampler.
        skip = self.skip_resampler(x)
        # Truncate / pad skip and residual to equal length
        target_len = min(skip.shape[-1], residual.shape[-1])
        skip = skip[..., :target_len]
        residual = residual[..., :target_len]

        out = mx.clip(residual + skip, -1.0, 1.0)[..., :output_length]
        return out.astype(input_dtype)

    def _compute_mel(self, audio: mx.array) -> mx.array:
        """Stereo mel computation.

        Input: ``(B, 2, T)`` waveform.
        Output: ``(B, 2, n_mel, T_frames)``.
        """
        B, C, T = audio.shape
        flat = audio.reshape(B * C, T)
        mel = self.mel_stft.mel_spectrogram(flat)  # (B*C, n_mel, T_frames)
        return mel.reshape(B, C, mel.shape[1], mel.shape[2])


__all__ = ["VocoderWithBWE"]
