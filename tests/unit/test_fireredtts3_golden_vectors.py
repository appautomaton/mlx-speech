from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.speaker import CAMPPlus, CAMPPlusConfig
from mlx_speech.models.fireredtts3.core import (
    FireRedTTS3Core,
    FireRedTTS3CoreConfig,
)
from mlx_speech.models.fireredtts3.redae import RedAE, RedAEConfig
from mlx_speech.models.fireredtts3.speaker import FireRedSpeakerEncoder
from scripts.audit.fireredtts3_golden_vectors import (
    MAX_PACK_BYTES,
    deterministic_weights,
    sha256_file,
    validate_fixture_pack,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/fireredtts3"


@pytest.fixture(autouse=True)
def _run_without_repository_model_paths(monkeypatch, tmp_path) -> None:
    """Allow MLX loads only from the committed fixture pack."""

    original_load = mx.load

    def fixture_only_load(path, *args, **kwargs):
        resolved = Path(path).resolve()
        assert resolved.is_relative_to(FIXTURE_DIR), (
            f"checkpoint-independent test tried to load {resolved}"
        )
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(mx, "load", fixture_only_load)
    monkeypatch.chdir(tmp_path)


def _manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _vectors() -> dict[str, np.ndarray]:
    with np.load(FIXTURE_DIR / "vectors.npz", allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _tolerance(manifest: dict, name: str) -> dict[str, float]:
    return manifest["tolerances"][name]


def test_golden_vector_pack_is_self_contained_bounded_and_hashed() -> None:
    manifest = validate_fixture_pack(FIXTURE_DIR)
    total_bytes = sum(path.stat().st_size for path in FIXTURE_DIR.iterdir())

    assert total_bytes <= MAX_PACK_BYTES
    assert set(manifest["files"]) == {"vectors.npz"}
    assert all(len(item["sha256"]) == 64 for item in manifest["files"].values())
    assert manifest["source_checkpoint"]["contract"] == {
        "patch_size": 4,
        "redae_dim": 64,
        "sample_rate": 24_000,
        "speaker_embedding_size": 512,
    }

    # The autouse guard runs this validation from an empty working directory and
    # permits MLX tensor loads only from the committed fixture pack.
    validate_fixture_pack(FIXTURE_DIR)


def test_micro_redae_matches_checkpoint_independent_vectors() -> None:
    manifest = _manifest()
    vectors = _vectors()
    model = RedAE(RedAEConfig(**manifest["micro_configs"]["redae"]))
    model.load_weights(deterministic_weights(model, phase=0.1), strict=True)
    model.eval()

    latents = model.encode(mx.array(vectors["redae_audio"]))
    decoded = model.decode(latents)
    mx.eval(latents, decoded)

    np.testing.assert_allclose(
        latents,
        vectors["redae_latents"],
        **_tolerance(manifest, "redae_latents"),
    )
    np.testing.assert_allclose(
        decoded,
        vectors["redae_decoded"],
        **_tolerance(manifest, "redae_decoded"),
    )


def test_micro_speaker_encoder_matches_checkpoint_independent_vectors() -> None:
    manifest = _manifest()
    vectors = _vectors()
    config = dict(manifest["micro_configs"]["speaker"])
    config["block_layers"] = tuple(config["block_layers"])
    model = CAMPPlus(CAMPPlusConfig(**config))
    model.load_weights(deterministic_weights(model, phase=0.3), strict=True)
    model.eval()
    encoder = FireRedSpeakerEncoder(model, max_audio_seconds=2.0)

    embedding = encoder(
        vectors["speaker_audio"],
        sample_rate=int(vectors["speaker_sample_rate"][0]),
    )
    mx.eval(embedding)
    np.testing.assert_allclose(
        embedding,
        vectors["speaker_embedding"],
        **_tolerance(manifest, "speaker_embedding"),
    )


def test_micro_core_matches_dit_and_two_patch_golden_vectors() -> None:
    manifest = _manifest()
    vectors = _vectors()
    core = FireRedTTS3Core(FireRedTTS3CoreConfig(**manifest["micro_configs"]["core"]))
    core.load_weights(deterministic_weights(core, phase=0.7), strict=True)
    core.eval()

    dit_output = core.dit(
        mx.array(vectors["dit_input"]),
        mx.array(vectors["dit_timestep"]),
    )
    generation = manifest["generation"]
    result = core.generate(
        speaker_embedding=mx.array(vectors["core_speaker"]),
        text_tokens=mx.array(vectors["core_tokens"]),
        prompt_latents=mx.array(vectors["core_prompt"]),
        flow_steps=generation["flow_steps"],
        guidance_scale=generation["guidance_scale"],
        stop_threshold=generation["stop_threshold"],
        max_generated_patches=generation["max_generated_patches"],
        seed=generation["seed"],
    )
    mx.eval(dit_output, result.latents)

    np.testing.assert_allclose(
        dit_output,
        vectors["dit_output"],
        **_tolerance(manifest, "dit_output"),
    )
    np.testing.assert_allclose(
        result.latents,
        vectors["core_latents"],
        **_tolerance(manifest, "core_latents"),
    )
    np.testing.assert_allclose(
        result.stop_scores,
        vectors["core_stop_scores"],
        **_tolerance(manifest, "core_stop_scores"),
    )
    np.testing.assert_array_equal(
        [result.generated_patches, result.cache_length, result.prompt_length],
        vectors["core_metadata"],
    )
    np.testing.assert_allclose(
        vectors["cosine_schedule"],
        [0.0, 1.0 - np.cos(np.pi / 4.0), 1.0],
        atol=1e-7,
        rtol=0.0,
    )


def test_fixture_manifest_does_not_mutate_when_validated() -> None:
    before = sha256_file(FIXTURE_DIR / "manifest.json")
    validate_fixture_pack(FIXTURE_DIR)
    assert sha256_file(FIXTURE_DIR / "manifest.json") == before
