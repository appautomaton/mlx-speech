"""Torch-absence guard for the DramaBox `denoise_ref=True` path (Slice 7).

The RE-USE / SEMamba enhancer is a pure-MLX port; the warm-server reference
used torch + Mamba CUDA kernels. This asserts that exercising the denoise path
never imports torch into the running process.

Gated on the converted RE-USE MLX weights (skipped if absent). It exercises the
real enhancer through the DramaBox helper `_denoise_reference_waveform`, which
is the exact transform `generate(denoise_ref=True)` runs on the reference, so a
torch import anywhere on that path would be caught here without needing the full
DramaBox/Gemma checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

REUSE_DIR = Path("models/reuse/mlx")

pytestmark = pytest.mark.skipif(
    not (REUSE_DIR / "model.safetensors").is_file(),
    reason="converted RE-USE MLX weights not present (run scripts/convert/reuse.py)",
)


def test_denoise_ref_path_imports_no_torch():
    """Running the denoise_ref transform must not import torch."""
    # Import the runtime modules first, then assert torch never appears after
    # actually running the denoise path. Importing here (not at module top)
    # keeps the check scoped to the work under test.
    from mlx_speech.generation.dramabox import _denoise_reference_waveform
    from mlx_speech.generation.reuse import REUSEEnhancer

    assert "torch" not in sys.modules, "torch was imported merely by importing the runtime"

    enhancer = REUSEEnhancer.from_dir(REUSE_DIR)

    # A [1, C, samples] reference, the shape generate() feeds the helper.
    sample_rate = 16_000
    t = mx.arange(sample_rate, dtype=mx.float32) / sample_rate
    mono = 0.3 * mx.sin(2 * mx.pi * 180.0 * t)
    waveform = mx.broadcast_to(mono[None, None, :], (1, 2, mono.shape[-1]))

    cleaned = _denoise_reference_waveform(enhancer, waveform, sample_rate)
    mx.eval(cleaned)

    assert cleaned.shape == waveform.shape
    assert mx.all(mx.isfinite(cleaned)).item()
    assert "torch" not in sys.modules, "torch was imported on the denoise_ref=True path"
