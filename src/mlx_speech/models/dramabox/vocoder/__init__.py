"""DramaBox vocoder stack — BigVGAN-v2 (main) + BigVGAN-v2 (BWE) + mel STFT.

The full chain decodes mel-spectrograms into 48 kHz stereo waveforms:

    mel [B, 2, T, 64]
        → main BigVGAN → 16 kHz wav
        → causal STFT (BWE's own mel front-end) → BWE BigVGAN → residual
        → sinc-resampled skip + residual → 48 kHz wav, clipped to [-1, 1]

All computation runs in fp32; bf16 weights are upcast at the boundary.
"""

from __future__ import annotations

from .checkpoint import load_vocoder_with_bwe_weights
from .vocoder import Vocoder
from .vocoder_with_bwe import VocoderWithBWE
from .mel_stft import MelSTFT

__all__ = [
    "Vocoder",
    "VocoderWithBWE",
    "MelSTFT",
    "load_vocoder_with_bwe_weights",
]
