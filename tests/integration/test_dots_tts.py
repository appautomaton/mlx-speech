from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.generation.dots_tts import DotsTTSGenerator


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("variant", ("soar", "mf"))
@pytest.mark.parametrize("artifact_class", ("base", "int8"))
@pytest.mark.parametrize("mode", ("continuation", "speaker_only"))
def test_dots_tts_four_artifacts_cover_both_clone_modes(
    variant: str,
    artifact_class: str,
    mode: str,
) -> None:
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("set RUN_LOCAL_INTEGRATION=1 for local checkpoint generation")
    model_dir = ROOT / "models/dots_tts" / variant / f"mlx-{artifact_class}"
    if not (model_dir / "core.safetensors").is_file():
        pytest.skip(f"local dots.tts {variant}/{artifact_class} is unavailable")
    corpus_lock = ROOT / "outputs/dots_tts/eval_corpus/manifest.lock.json"
    if not corpus_lock.is_file():
        pytest.skip("materialize the dots.tts multilingual eval corpus")
    reference = json.loads(corpus_lock.read_text(encoding="utf-8"))["references"][0]

    generator = DotsTTSGenerator.from_dir(model_dir)
    result = generator.synthesize(
        reference["target_text"],
        reference_audio=ROOT / reference["path"],
        reference_text=(
            reference["reference_text"] if mode == "continuation" else None
        ),
        language=reference["language"],
        max_audio_patches=128 if mode == "continuation" else 1,
        solver_steps=1,
        seed=37,
        eos_threshold=0.0,
    )
    assert result.sample_rate == 48_000
    assert result.num_patches == 1
    assert result.waveform.ndim == 1
    assert int(result.waveform.size) > 0
    assert bool(mx.all(mx.isfinite(result.waveform)).item())
    assert bool(mx.any(mx.abs(result.waveform) > 0).item())
    del generator, result
    gc.collect()
    mx.clear_cache()
