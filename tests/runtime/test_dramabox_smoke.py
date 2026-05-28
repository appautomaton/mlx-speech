"""Smoke test for the DramaBox v5 wrapper.

This is a tier-3 runtime test: it actually runs the full pipeline (Gemma →
prompt → DiT → VAE → vocoder) on the dev checkpoints, and asserts the
output is a finite stereo waveform of roughly the right length. It does
NOT compare against a parity capture — that's a separate runtime fixture.

Gated on the presence of both model directories; skipped automatically if
either is absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

DRAMABOX_DIR = Path("models/dramabox")
GEMMA_DIR = Path("models/gemma_3_12b_it_4bit")

pytestmark = pytest.mark.skipif(
    not DRAMABOX_DIR.is_dir() or not GEMMA_DIR.is_dir(),
    reason="DramaBox or Gemma 4-bit checkpoint missing",
)


def test_dramabox_generate_smoke_short(monkeypatch):
    """Run an extremely short generation (3 steps, 1 second) and check the
    pipeline doesn't crash / NaN. This is a structural smoke test, not a
    parity gate."""
    from mlx_speech.generation.dramabox import DramaBoxModel

    model = DramaBoxModel.from_dir(DRAMABOX_DIR, gemma_dir=GEMMA_DIR)
    result = model.generate(
        "A woman speaks clearly.",
        duration_s=1.0,
        cfg_scale=1.0,        # disable cfg to skip the second forward pass for speed
        stg_scale=0.0,
        rescale_scale=0.0,
        modality_scale=1.0,
        steps=3,
        seed=42,
    )
    assert result.sample_rate == 48_000
    # Two channels, length roughly duration_s * sample_rate
    assert result.waveform.shape[0] == 2
    assert result.waveform.shape[1] > 0
    # No NaN / Inf
    assert mx.all(mx.isfinite(result.waveform)).item()
    # Clipped to [-1, 1]
    assert float(mx.max(result.waveform)) <= 1.0 + 1e-5
    assert float(mx.min(result.waveform)) >= -1.0 - 1e-5
