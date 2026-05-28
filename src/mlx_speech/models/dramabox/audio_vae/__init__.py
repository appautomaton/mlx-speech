"""DramaBox AudioVAE — mel spectrogram autoencoder.

Submodules:
- `config.py` — VAE config dataclass
- `causal_conv_2d.py` — causal `Conv2d` along the height (time) axis
- `pixel_norm.py` — parameter-less PixelNorm
- `resnet.py` — VAE ResnetBlock
- `encoder.py` / `decoder.py` — the encoder/decoder towers
- `per_channel_statistics.py` — train-time per-channel mean/std
- `audio_processor.py` — waveform → mel STFT front-end (needed only
  for voice-reference encoding which lands in Stage 7)
- `reference_prep.py` — voice-reference WAV → encoder-ready stereo waveform
- `model.py` — `AudioVAE` container with `encode`, `decode`, `unnormalize`
- `checkpoint.py` — load from audio-components shard
"""

from __future__ import annotations

from .audio_processor import AudioProcessor
from .checkpoint import load_audio_vae_weights
from .config import AudioVAEConfig
from .model import AudioVAE
from .per_channel_statistics import PerChannelStatistics
from .reference_prep import ReferenceAudio, prepare_reference_audio

__all__ = [
    "AudioProcessor",
    "AudioVAE",
    "AudioVAEConfig",
    "PerChannelStatistics",
    "ReferenceAudio",
    "load_audio_vae_weights",
    "prepare_reference_audio",
]
