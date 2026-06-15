"""Runtime A/B check that STG actually perturbs DramaBox generation.

Tier-3 runtime test: runs the full pipeline twice on the dev checkpoints with
STG off vs on (cfg fixed at 1.0 so the *only* difference is the perturbed
pass), and asserts both outputs are finite stereo 48 kHz and that they differ.
Guards against a silent no-op regression in the STG wiring.

Gated on both model directories; skipped automatically if either is absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

DRAMABOX_DIR = Path("models/dramabox/mlx-bf16")
GEMMA_DIR = Path("models/gemma_3_12b_it_backbone/mlx-4bit")

pytestmark = pytest.mark.skipif(
    not DRAMABOX_DIR.is_dir() or not GEMMA_DIR.is_dir(),
    reason="DramaBox or Gemma 4-bit checkpoint missing",
)


def test_stg_changes_output_vs_cfg_only():
    from mlx_speech.generation.dramabox import DramaBoxModel

    model = DramaBoxModel.from_dir(DRAMABOX_DIR, gemma_dir=GEMMA_DIR)
    kwargs = dict(
        duration_s=1.0,
        cfg_scale=1.0,        # isolate STG: no uncond pass, so the only delta is ptb
        rescale_scale=0.0,
        modality_scale=1.0,
        steps=4,
        seed=42,
    )
    cfg_only = model.generate("A woman speaks clearly.", stg_scale=0.0, **kwargs)
    with_stg = model.generate("A woman speaks clearly.", stg_scale=1.5, **kwargs)

    # Both well-formed stereo 48 kHz.
    for out in (cfg_only, with_stg):
        assert out.sample_rate == 48_000
        assert out.waveform.shape[0] == 2
        assert out.waveform.shape[1] > 0
        assert mx.all(mx.isfinite(out.waveform)).item()
        assert float(mx.max(out.waveform)) <= 1.0 + 1e-5
        assert float(mx.min(out.waveform)) >= -1.0 - 1e-5

    # STG must measurably change the result (no silent fallback).
    assert with_stg.waveform.shape == cfg_only.waveform.shape
    delta = float(mx.max(mx.abs(with_stg.waveform - cfg_only.waveform)).item())
    assert delta > 1e-4, f"STG produced no change (delta={delta})"
