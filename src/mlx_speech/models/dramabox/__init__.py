"""DramaBox MLX runtime — flow-matching diffusion TTS, audio-only.

Submodules:
- `prompt/` — text → `a_ctx` (FeatureExtractorV2 + aggregate + connector)
- `ltx/` — shared LTX-2 primitives (RoPE, attention, FFN) reused by Stage 6
"""

from __future__ import annotations
