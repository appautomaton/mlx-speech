from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech import tts


ROOT = Path(__file__).parents[2]
REMOTE_REVISION = "5dde9ded6c577a84a71b5ee9dafebfa53188d6d6"
REMOTE_CASES = (
    ("dots-tts-soar-base", "soar/mlx-base", "batch"),
    ("dots-tts-soar", "soar/mlx-int8", "stream"),
    ("dots-tts-mf-base", "mf/mlx-base", "batch"),
    ("dots-tts-mf", "mf/mlx-int8", "stream"),
)


@pytest.mark.parametrize(("alias", "artifact_path", "sink"), REMOTE_CASES)
def test_remote_alias_isolated_cache_and_public_waveform_sink(
    alias: str,
    artifact_path: str,
    sink: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("set RUN_LOCAL_INTEGRATION=1 for remote dots.tts generation")
    corpus_lock = ROOT / "outputs/dots_tts/eval_corpus/manifest.lock.json"
    if not corpus_lock.is_file():
        pytest.skip("materialize the dots.tts multilingual eval corpus")

    cache = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(cache.parent / "hf-home"))
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    model = tts.load(alias, revision=REMOTE_REVISION)
    model_dir = model._generator.components.layout.model_dir
    snapshot_root = model_dir.parents[1]
    assert model_dir.relative_to(snapshot_root).as_posix() == artifact_path
    assert (snapshot_root / "README.md").is_file()
    safetensors = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*.safetensors")
    }
    assert safetensors == {
        f"{artifact_path}/core.safetensors",
        f"{artifact_path}/latent_stats.safetensors",
        f"{artifact_path}/speaker.safetensors",
        f"{artifact_path}/vocoder.safetensors",
    }

    reference = json.loads(corpus_lock.read_text(encoding="utf-8"))["references"][0]
    kwargs = {
        "reference_audio": ROOT / reference["path"],
        "reference_text": reference["reference_text"],
        "language": reference["language"],
        "max_audio_patches": 128,
        "solver_steps": 1,
        "seed": 37,
        "eos_threshold": 0.0,
    }
    if sink == "batch":
        result = model.generate(reference["target_text"], **kwargs)
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
