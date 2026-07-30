from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech import tts


ROOT = Path(__file__).parents[2]
REMOTE_REVISION = "0af7ad2f837278b364902500d086553f1586ce9a"
REMOTE_CASES = (
    ("dots-tts-soar-base", "soar/mlx-base"),
    ("dots-tts-soar", "soar/mlx-int8"),
    ("dots-tts-mf-base", "mf/mlx-base"),
    ("dots-tts-mf", "mf/mlx-int8"),
)


@pytest.mark.parametrize(("alias", "artifact_path"), REMOTE_CASES)
def test_remote_alias_isolated_cache_and_continuation_waveform(
    alias: str,
    artifact_path: str,
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
    result = model.generate(
        reference["target_text"],
        reference_audio=ROOT / reference["path"],
        reference_text=reference["reference_text"],
        language=reference["language"],
        max_audio_patches=128,
        solver_steps=1,
        seed=37,
        eos_threshold=0.0,
    )
    assert result.sample_rate == 48_000
    assert result.waveform.ndim == 1
    assert int(result.waveform.size) > 0
    assert bool(mx.all(mx.isfinite(result.waveform)).item())
    assert bool(mx.any(mx.abs(result.waveform) > 0).item())

    del model, result
    gc.collect()
    mx.clear_cache()
