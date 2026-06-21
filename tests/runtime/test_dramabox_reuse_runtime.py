"""Runtime A-B for DramaBox `denoise_ref=True` (Slice 7).

Tier-3 runtime test: runs the full DramaBox pipeline twice (raw reference vs
RE-USE-cleaned reference) on the dev checkpoints and asserts the cleaning
actually changes the conditioning. Slow by design (two short generations).

Gated on the presence of all three weight sets (DramaBox, Gemma 4-bit, and the
converted RE-USE MLX weights); skipped automatically if any is absent. The
RE-USE weights are injected locally via `reuse_path_or_repo` so the test never
hits the network.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

DRAMABOX_DIR = Path("models/dramabox/mlx-bf16")
GEMMA_DIR = Path("models/gemma_3_12b_it_backbone/mlx-4bit")
REUSE_DIR = Path("models/reuse/mlx")

pytestmark = pytest.mark.skipif(
    not DRAMABOX_DIR.is_dir()
    or not GEMMA_DIR.is_dir()
    or not (REUSE_DIR / "model.safetensors").is_file(),
    reason="DramaBox, Gemma 4-bit, or converted RE-USE weights missing",
)

# Short + cheap settings: cfg/stg/rescale off, few steps, 1 s clip.
_GEN_KWARGS = dict(
    duration_s=1.0,
    cfg_scale=1.0,
    stg_scale=0.0,
    rescale_scale=0.0,
    modality_scale=1.0,
    steps=3,
    seed=123,
)


def _write_noisy_ref(tmp_path: Path) -> Path:
    """A noisy speech-like reference: a tone plus broadband noise so RE-USE has
    something to remove (and thus measurably changes the conditioning)."""
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate  # 1 s
    rng = np.random.default_rng(0)
    tone = 0.3 * np.sin(2 * np.pi * 180.0 * t)
    noise = 0.15 * rng.standard_normal(t.shape[0]).astype(np.float32)
    ref = (tone + noise).astype(np.float32)
    ref_path = tmp_path / "noisy_voice_ref.wav"
    sf.write(ref_path, ref, sample_rate)
    return ref_path


def test_denoise_ref_changes_conditioning(tmp_path):
    """denoise_ref=True yields finite 48 kHz stereo that differs from False."""
    from mlx_speech.generation.dramabox import DramaBoxModel

    ref_path = _write_noisy_ref(tmp_path)

    model = DramaBoxModel.from_dir(
        DRAMABOX_DIR, gemma_dir=GEMMA_DIR, reuse_path_or_repo=REUSE_DIR
    )

    raw = model.generate("A woman speaks clearly.", voice_ref=ref_path, denoise_ref=False, **_GEN_KWARGS)
    cleaned = model.generate("A woman speaks clearly.", voice_ref=ref_path, denoise_ref=True, **_GEN_KWARGS)

    # Both finite 48 kHz stereo, same shape.
    for result in (raw, cleaned):
        assert result.sample_rate == 48_000
        assert result.waveform.shape[0] == 2
        assert result.waveform.shape[1] > 0
        assert mx.all(mx.isfinite(result.waveform)).item()
        assert float(mx.max(result.waveform)) <= 1.0 + 1e-5
        assert float(mx.min(result.waveform)) >= -1.0 - 1e-5

    assert cleaned.waveform.shape == raw.waveform.shape
    # RE-USE cleaning changed the reference, so the conditioned output differs.
    delta = float(mx.max(mx.abs(cleaned.waveform - raw.waveform)).item())
    assert delta > 1e-5, "denoise_ref=True did not change the output"


def test_cleaned_reference_is_cached(tmp_path):
    """A second denoise_ref=True call with the same path reuses the cached
    cleaned reference rather than re-denoising."""
    from mlx_speech.generation.dramabox import DramaBoxModel

    ref_path = _write_noisy_ref(tmp_path)

    model = DramaBoxModel.from_dir(
        DRAMABOX_DIR, gemma_dir=GEMMA_DIR, reuse_path_or_repo=REUSE_DIR
    )

    model.generate("A woman speaks clearly.", voice_ref=ref_path, denoise_ref=True, **_GEN_KWARGS)
    cache_key = (str(ref_path), 16_000)
    assert cache_key in model._ref_denoise_cache
    cached_first = model._ref_denoise_cache[cache_key]

    model.generate("A woman speaks clearly.", voice_ref=ref_path, denoise_ref=True, **_GEN_KWARGS)
    # Same object: not re-denoised on the second call.
    assert model._ref_denoise_cache[cache_key] is cached_first
