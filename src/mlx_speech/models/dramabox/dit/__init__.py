"""DramaBox audio-only DiT — the 3.3B-param flow-matching transformer.

Audio-only LTX-2 model. 48 transformer blocks, each with:
- self-attention (`audio_attn1`) on the patchified latent
- cross-attention (`audio_attn2`) to the prompt's `a_ctx` features
- feed-forward (`audio_ff`)
- AdaLN modulation derived from the timestep (`audio_adaln_single`)
- prompt-AdaLN modulation of the cross-attn context (`audio_prompt_adaln_single`)

Public submodules:
- `config.py` — `DiTConfig`
- `timestep.py` — PixArt-style timestep embedding (sinusoidal → MLP)
- `block.py` — `LTXBlock`
- `model.py` — `LTXModel` (top-level forward returning velocity)
- `checkpoint.py` — loader for `dramabox-dit-v1.safetensors`
"""

from __future__ import annotations

from .checkpoint import load_dit_weights
from .config import DiTConfig
from .model import LTXModel

__all__ = ["DiTConfig", "LTXModel", "load_dit_weights"]
