#!/usr/bin/env python3
"""Capture checkpoint-independent FireRedTTS3 regression vectors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_speech.generation.fireredtts3 import cosine_time_schedule
from mlx_speech.models.dots_tts.speaker import CAMPPlus, CAMPPlusConfig
from mlx_speech.models.fireredtts3.config import FireRedTTS3Config, TOKENIZER_FILES
from mlx_speech.models.fireredtts3.core import (
    FireRedTTS3Core,
    FireRedTTS3CoreConfig,
)
from mlx_speech.models.fireredtts3.redae import RedAE, RedAEConfig
from mlx_speech.models.fireredtts3.speaker import FireRedSpeakerEncoder


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = Path(__file__).resolve()
DEFAULT_MODEL_DIR = ROOT / "models/firered/firered_tts3/mlx-bf16"
DEFAULT_FIXTURE_DIR = ROOT / "tests/fixtures/fireredtts3"
SCHEMA_VERSION = 1
MAX_PACK_BYTES = 8 * 1024 * 1024

RED_AE_CONFIG = RedAEConfig(
    audio_patch_size=4,
    audio_sample_rate=24,
    bottleneck_dim=4,
    head_dim=4,
    enc_hidden_size=8,
    enc_intermediate_size=16,
    enc_num_hidden_layers=1,
    enc_max_position_embeddings=64,
    enc_num_attention_heads=2,
    enc_num_key_value_heads=1,
    enc_extra_downsample_rate=2,
    enc_downsample_num_hidden_layers=1,
    dec_hidden_size=8,
    dec_intermediate_size=16,
    dec_num_hidden_layers=1,
    dec_max_position_embeddings=64,
    dec_num_attention_heads=2,
    dec_num_key_value_heads=1,
    vocab_size=32,
    rope_theta=10_000.0,
)

SPEAKER_CONFIG = CAMPPlusConfig(
    feature_dim=80,
    embedding_size=12,
    growth_rate=2,
    bottleneck_size=2,
    initial_channels=4,
    block_layers=(1, 1, 1),
)

CORE_CONFIG = FireRedTTS3CoreConfig(
    redae_dim=4,
    num_history_patches=2,
    spk_in_dim=6,
    patch_size=2,
    patch_encoder_hidden_size=8,
    patch_encoder_mlp_ratio=2,
    patch_encoder_depth=1,
    patch_encoder_num_heads=2,
    dit_mlp_ratio=2,
    dit_depth=1,
    dit_num_heads=2,
    dit_hidden_size=8,
    qwen_hidden_size=8,
    qwen_intermediate_size=16,
    qwen_num_hidden_layers=1,
    qwen_num_attention_heads=2,
    qwen_num_key_value_heads=1,
    qwen_head_dim=4,
    qwen_vocab_size=32,
    qwen_max_position_embeddings=64,
    qwen_rope_theta=10_000.0,
)

TOLERANCES = {
    "redae_latents": {"atol": 2e-5, "rtol": 2e-5},
    "redae_decoded": {"atol": 2e-5, "rtol": 2e-5},
    "speaker_embedding": {"atol": 2e-4, "rtol": 2e-4},
    "dit_output": {"atol": 2e-5, "rtol": 2e-5},
    "core_latents": {"atol": 2e-5, "rtol": 2e-5},
    "core_stop_scores": {"atol": 2e-5, "rtol": 2e-5},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def deterministic_weights(module, *, phase: float) -> list[tuple[str, mx.array]]:
    """Build stable tiny-model parameters without committing weight files."""

    flattened = tree_flatten(module.parameters(), destination={})
    weights: list[tuple[str, mx.array]] = []
    for index, (name, parameter) in enumerate(sorted(flattened.items())):
        values = mx.arange(parameter.size, dtype=mx.float32).reshape(parameter.shape)
        values = mx.sin(values + phase + float(index + 1)) * 0.04
        if name.endswith("running_var") or (
            name.endswith("weight") and ("norm" in name or ".bn" in name)
        ):
            values = values + 1.0
        weights.append((name, values.astype(parameter.dtype)))
    return weights


def build_micro_models() -> tuple[RedAE, FireRedSpeakerEncoder, FireRedTTS3Core]:
    mx.random.seed(101)
    redae = RedAE(RED_AE_CONFIG)
    redae.load_weights(deterministic_weights(redae, phase=0.1), strict=True)
    redae.eval()
    mx.random.seed(103)
    speaker_model = CAMPPlus(SPEAKER_CONFIG)
    speaker_model.load_weights(
        deterministic_weights(speaker_model, phase=0.3), strict=True
    )
    speaker_model.eval()
    speaker = FireRedSpeakerEncoder(speaker_model, max_audio_seconds=2.0)
    mx.random.seed(107)
    core = FireRedTTS3Core(CORE_CONFIG)
    core.load_weights(deterministic_weights(core, phase=0.7), strict=True)
    core.eval()
    return redae, speaker, core


def capture_vectors(
    redae: RedAE,
    speaker: FireRedSpeakerEncoder,
    core: FireRedTTS3Core,
) -> dict[str, np.ndarray]:
    redae_audio = mx.linspace(-0.1, 0.1, 32, dtype=mx.float32)[None]
    redae_latents = redae.encode(redae_audio)
    redae_decoded = redae.decode(redae_latents)

    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    speaker_audio = (
        np.linspace(0.4, 1.0, sample_rate, dtype=np.float32)
        * (
            0.12 * np.sin(2.0 * np.pi * 180.0 * time)
            + 0.03 * np.sin(2.0 * np.pi * 360.0 * time + 0.2)
        )
    ).astype(np.float32)
    speaker_embedding = speaker(speaker_audio, sample_rate=sample_rate)

    core_speaker = mx.linspace(-0.2, 0.2, 6, dtype=mx.float32)[None]
    core_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    core_prompt = mx.linspace(-0.1, 0.1, 32, dtype=mx.float32).reshape(1, 8, 4)
    dit_input = mx.linspace(-0.3, 0.3, 216, dtype=mx.float32).reshape(2, 6, 18)
    dit_timestep = mx.array([0.25, 0.25], dtype=mx.float32)
    dit_output = core.dit(dit_input, dit_timestep)
    core_result = core.generate(
        speaker_embedding=core_speaker,
        text_tokens=core_tokens,
        prompt_latents=core_prompt,
        flow_steps=2,
        guidance_scale=1.0,
        stop_threshold=2.0,
        max_generated_patches=2,
        seed=17,
    )
    schedule = cosine_time_schedule(2)
    mx.eval(
        redae_latents,
        redae_decoded,
        speaker_embedding,
        dit_output,
        core_result.latents,
        schedule,
    )
    return {
        "redae_audio": np.asarray(redae_audio, dtype=np.float32),
        "redae_latents": np.asarray(redae_latents, dtype=np.float32),
        "redae_decoded": np.asarray(redae_decoded, dtype=np.float32),
        "speaker_audio": speaker_audio,
        "speaker_sample_rate": np.asarray([sample_rate], dtype=np.int32),
        "speaker_embedding": np.asarray(speaker_embedding, dtype=np.float32),
        "core_speaker": np.asarray(core_speaker, dtype=np.float32),
        "core_tokens": np.asarray(core_tokens, dtype=np.int32),
        "core_prompt": np.asarray(core_prompt, dtype=np.float32),
        "dit_input": np.asarray(dit_input, dtype=np.float32),
        "dit_timestep": np.asarray(dit_timestep, dtype=np.float32),
        "dit_output": np.asarray(dit_output, dtype=np.float32),
        "core_latents": np.asarray(core_result.latents, dtype=np.float32),
        "core_stop_scores": np.asarray(core_result.stop_scores, dtype=np.float32),
        "core_metadata": np.asarray(
            [
                core_result.generated_patches,
                core_result.cache_length,
                core_result.prompt_length,
            ],
            dtype=np.int32,
        ),
        "cosine_schedule": np.asarray(schedule, dtype=np.float32),
    }


def _array_inventory(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": int(value.nbytes),
        }
        for name, value in sorted(vectors.items())
    }


def _source_inventory(model_dir: Path) -> dict[str, object]:
    artifact = FireRedTTS3Config.from_dir(model_dir)
    names = ["config.json", *artifact.files.values(), *TOKENIZER_FILES]
    files: dict[str, object] = {}
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"FireRedTTS3 source artifact is missing: {path}")
        digest = sha256_file(path)
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
    contract = {
        "sample_rate": artifact.sample_rate,
        "patch_size": int(artifact.core["patch_size"]),
        "redae_dim": int(artifact.core["redae_dim"]),
        "speaker_embedding_size": int(artifact.speaker["embedding_size"]),
    }
    return {"files": files, "contract": contract}


def capture_fixture_pack(output_dir: Path, model_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace fixture pack: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _source_inventory(model_dir)
    # Match the test backend even when capture runs on a Mac with a Metal GPU.
    with mx.stream(mx.cpu):
        redae, speaker, core = build_micro_models()
        vectors = capture_vectors(redae, speaker, core)
    vectors_path = output_dir / "vectors.npz"
    np.savez_compressed(vectors_path, **vectors)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": "FireRedTTS3 checkpoint-independent golden test vectors",
        "description": (
            "Self-contained deterministic micro-model inputs and expected outputs "
            "for regression testing without production model weights."
        ),
        "capture": {
            "device": "cpu",
            "git_commit": _git_head(),
            "script_sha256": sha256_file(CAPTURE_SCRIPT),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__,
            "command": (
                "python scripts/audit/fireredtts3_golden_vectors.py capture "
                "--model-dir models/firered/firered_tts3/mlx-bf16 "
                "--output-dir tests/fixtures/fireredtts3"
            ),
        },
        "source_checkpoint": source,
        "micro_configs": {
            "redae": asdict(RED_AE_CONFIG),
            "speaker": asdict(SPEAKER_CONFIG),
            "core": asdict(CORE_CONFIG),
        },
        "generation": {
            "flow_steps": 2,
            "guidance_scale": 1.0,
            "stop_threshold": 2.0,
            "max_generated_patches": 2,
            "seed": 17,
        },
        "tolerances": TOLERANCES,
        "arrays": _array_inventory(vectors),
        "files": {
            vectors_path.name: {
                "sha256": sha256_file(vectors_path),
                "bytes": vectors_path.stat().st_size,
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_fixture_pack(output_dir)
    return output_dir


def validate_fixture_pack(fixture_dir: Path) -> dict[str, object]:
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported FireRedTTS3 fixture schema")
    if manifest["capture"].get("device") != "cpu":
        raise ValueError("FireRedTTS3 golden vectors require CPU capture")
    if manifest["capture"].get("script_sha256") != sha256_file(CAPTURE_SCRIPT):
        raise ValueError("fixture capture script provenance mismatch")
    expected_files = set(manifest["files"])
    actual_files = {
        path.name for path in fixture_dir.iterdir() if path.name != "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError(
            f"fixture files differ: expected={sorted(expected_files)}, "
            f"actual={sorted(actual_files)}"
        )
    total_bytes = manifest_path.stat().st_size
    for name, expected in manifest["files"].items():
        path = fixture_dir / name
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"fixture SHA-256 mismatch: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"fixture size mismatch: {name}")
        total_bytes += path.stat().st_size
    if total_bytes > MAX_PACK_BYTES:
        raise ValueError(f"fixture pack exceeds {MAX_PACK_BYTES} bytes")
    with np.load(fixture_dir / "vectors.npz", allow_pickle=False) as payload:
        if set(payload.files) != set(manifest["arrays"]):
            raise ValueError("fixture vector inventory differs from manifest")
        for name in payload.files:
            value = payload[name]
            expected = manifest["arrays"][name]
            if list(value.shape) != expected["shape"]:
                raise ValueError(f"fixture shape mismatch: {name}")
            if str(value.dtype) != expected["dtype"]:
                raise ValueError(f"fixture dtype mismatch: {name}")
            if int(value.nbytes) != expected["bytes"]:
                raise ValueError(f"fixture byte count mismatch: {name}")
    return manifest


def compare_fixture_packs(expected_dir: Path, actual_dir: Path) -> None:
    expected_manifest = validate_fixture_pack(expected_dir)
    actual_manifest = validate_fixture_pack(actual_dir)
    for section in ("micro_configs", "generation", "source_checkpoint"):
        if expected_manifest[section] != actual_manifest[section]:
            raise ValueError(f"fixture {section} differs")
    with np.load(expected_dir / "vectors.npz", allow_pickle=False) as expected:
        with np.load(actual_dir / "vectors.npz", allow_pickle=False) as actual:
            for name in expected.files:
                tolerance = TOLERANCES.get(name, {"atol": 0.0, "rtol": 0.0})
                np.testing.assert_allclose(
                    actual[name],
                    expected[name],
                    atol=tolerance["atol"],
                    rtol=tolerance["rtol"],
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    capture.add_argument("--output-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    regenerate = subparsers.add_parser("regenerate")
    regenerate.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    regenerate.add_argument("--compare", type=Path, default=DEFAULT_FIXTURE_DIR)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "capture":
        output = capture_fixture_pack(args.output_dir, args.model_dir)
        print(f"Captured FireRedTTS3 golden vectors in {output}")
        return
    if args.command == "regenerate":
        with tempfile.TemporaryDirectory(prefix="fireredtts3-golden-") as temp:
            regenerated = capture_fixture_pack(Path(temp) / "fixtures", args.model_dir)
            compare_fixture_packs(args.compare, regenerated)
        print(f"Regenerated vectors match {args.compare}")
        return
    validate_fixture_pack(args.fixture_dir)
    print(f"Validated FireRedTTS3 golden vectors in {args.fixture_dir}")


if __name__ == "__main__":
    main()
