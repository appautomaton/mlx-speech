from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech import tts


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

    model = tts.load(str(model_dir))
    kwargs = {
        "reference_audio": ROOT / reference["path"],
        "language": reference["language"],
        "max_audio_patches": 128 if mode == "continuation" else 1,
        "solver_steps": 1,
        "seed": 37,
        "eos_threshold": 0.0,
    }
    if mode == "continuation":
        result = model.generate(
            reference["target_text"],
            reference_text=reference["reference_text"],
            **kwargs,
        )
        waveform = result.waveform
        sample_rate = result.sample_rate
    else:
        chunks = list(model.generate_stream(reference["target_text"], **kwargs))
        assert chunks
        assert all(chunk.sample_rate == 48_000 for chunk in chunks)
        waveform = mx.concatenate([chunk.waveform for chunk in chunks])
        sample_rate = chunks[0].sample_rate
    assert sample_rate == 48_000
    assert waveform.ndim == 1
    assert int(waveform.size) > 0
    assert bool(mx.all(mx.isfinite(waveform)).item())
    assert bool(mx.any(mx.abs(waveform) > 0).item())
    del model, waveform
    gc.collect()
    mx.clear_cache()
