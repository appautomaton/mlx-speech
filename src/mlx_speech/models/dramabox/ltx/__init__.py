"""LTX-2 primitives shared between the embeddings connector and the audio DiT.

Files:
- `rope.py` — split/interleaved RoPE with NumPy-fp64 frequency grid
- `attention.py` — multi-head attention with optional per-head gating
- `feed_forward.py` — GELU-approx + linear FFN (no gating)
- `rms_norm.py` — standard RMSNorm (used by connector blocks)
"""

from __future__ import annotations
