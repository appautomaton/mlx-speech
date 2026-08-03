#!/usr/bin/env python3
"""Upload appautomaton MLX model artifacts to Hugging Face.

Usage:
    # Upload one model
    python scripts/hugging_face/upload.py vibevoice

    # Upload multiple models
    python scripts/hugging_face/upload.py vibevoice openmoss-ttsd

    # List available targets
    python scripts/hugging_face/upload.py --list

    # Upload all
    python scripts/hugging_face/upload.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Registry: alias -> (hf_repo_id, local_path, use_large_folder)
#
# use_large_folder=True  → hf upload-large-folder  (chunked, resumable)
# use_large_folder=False → hf upload               (simple single-path)
# ---------------------------------------------------------------------------
MODELS: dict[str, tuple[str, str, bool]] = {
    "cohere-asr": (
        "appautomaton/cohere-asr-mlx",
        "models/cohere/cohere_transcribe/mlx-int8",
        False,
    ),
    # bf16 weights are already published; re-publish only to refresh the card
    # (push the card as README.md — see scripts/hugging_face/model_cards/).
    "qwen3-asr-1.7b-int8": (
        "appautomaton/qwen3-asr-1.7b-int8-mlx",
        "models/qwen3_asr_1_7b/mlx-int8",
        True,
    ),
    "nemotron-asr-streaming-int8": (
        "appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx",
        "models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-int8",
        True,
    ),
    "openmoss-audio-tokenizer": (
        "appautomaton/openmoss-audio-tokenizer-mlx",
        "models/openmoss/moss_audio_tokenizer/mlx-int8",
        False,
    ),
    "openmoss-tts-local": (
        "appautomaton/openmoss-tts-local-mlx",
        "models/openmoss/moss_tts_local/mlx-int8",
        False,
    ),
    "openmoss-ttsd": (
        "appautomaton/openmoss-ttsd-mlx",
        "models/openmoss/moss_ttsd/mlx-int8",
        False,
    ),
    "openmoss-sound-effect": (
        "appautomaton/openmoss-sound-effect-mlx",
        "models/openmoss/moss_sound_effect/mlx-4bit",
        True,
    ),
    "vibevoice": (
        "appautomaton/vibevoice-mlx",
        "models/vibevoice/mlx-int8",
        True,
    ),
    "dramabox": (
        "appautomaton/dramabox-tts-3.3b-bf16-mlx",
        "models/dramabox/mlx-bf16",
        True,
    ),
    "gemma-3-backbone": (
        "appautomaton/gemma-3-12b-it-backbone-4bit-mlx",
        "models/gemma_3_12b_it_backbone/mlx-4bit",
        True,
    ),
    "reuse": (
        "appautomaton/re-use-semamba-mlx",
        "models/reuse/mlx",
        True,
    ),
}

DOTS_TTS_REPO_ID = "appautomaton/dots-tts-mlx"
DOTS_TTS_ARTIFACTS = (
    ("soar", "base", "soar/mlx-base"),
    ("soar", "int8", "soar/mlx-int8"),
    ("mf", "base", "mf/mlx-base"),
    ("mf", "int8", "mf/mlx-int8"),
)
DOTS_TTS_INCLUDE_PATTERNS = tuple(
    f"{remote_path}/**" for _, _, remote_path in DOTS_TTS_ARTIFACTS
)
DOTS_TTS_WEIGHT_FILES = (
    "core.safetensors",
    "vocoder.safetensors",
    "speaker.safetensors",
    "latent_stats.safetensors",
)
DOTS_TTS_REQUIRED_FILES = {
    "config.json",
    "llm_config.json",
    "mlx_config.json",
    *DOTS_TTS_WEIGHT_FILES,
}
DOTS_TTS_TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
DOTS_TTS_SOURCE_REVISIONS = {
    "soar": "e3520f75254d0020a0406db31c51a79d00d22d55",
    "mf": "25c53fb462e57087e52237daa5ea30df1c5cc328",
}
SPECIAL_TARGETS = ("dots-tts",)


def _is_usable_script(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        with path.open("rb") as handle:
            first_line = handle.readline(4096).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return True
    if not first_line.startswith("#!"):
        return True
    interpreter = first_line[2:].strip().split(maxsplit=1)[0]
    return not interpreter.startswith("/") or Path(interpreter).exists()


def _resolve_hf() -> str:
    for directory in os.get_exec_path():
        candidate = Path(directory) / "hf"
        if _is_usable_script(candidate):
            return str(candidate)
    raise FileNotFoundError(
        "No usable hf CLI was found on PATH; install huggingface_hub or repair "
        "the selected environment entrypoint"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dots_tts_artifact_digest(path: Path) -> str:
    inventory: dict[str, dict[str, str | int]] = {}
    for name in (*DOTS_TTS_WEIGHT_FILES, "mlx_config.json"):
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f"Missing dots.tts release file: {file}")
        inventory[name] = {
            "bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _require_files(path: Path, names: set[str]) -> None:
    missing = sorted(name for name in names if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing dots.tts release files under {path}: {', '.join(missing)}"
        )


def _require_exact_artifact_layout(path: Path) -> None:
    _require_files(path, DOTS_TTS_REQUIRED_FILES)
    tokenizer = path / "tokenizer"
    _require_files(tokenizer, DOTS_TTS_TOKENIZER_FILES)

    top_files = {item.name for item in path.iterdir() if item.is_file()}
    top_directories = {item.name for item in path.iterdir() if item.is_dir()}
    tokenizer_files = {item.name for item in tokenizer.iterdir() if item.is_file()}
    tokenizer_directories = {item.name for item in tokenizer.iterdir() if item.is_dir()}
    unexpected = {
        "top_files": sorted(top_files - DOTS_TTS_REQUIRED_FILES),
        "top_directories": sorted(top_directories - {"tokenizer"}),
        "tokenizer_files": sorted(tokenizer_files - DOTS_TTS_TOKENIZER_FILES),
        "tokenizer_directories": sorted(tokenizer_directories),
    }
    if any(unexpected.values()):
        raise ValueError(
            f"Unexpected dots.tts release content under {path}: {unexpected}"
        )


def _dots_tts_release_manifest(root: Path) -> dict[str, Any]:
    model_root = root / "models" / "dots_tts"
    card = (
        root
        / "scripts"
        / "hugging_face"
        / "model_cards"
        / "appautomaton"
        / "dots-tts-mlx.md"
    )
    report_path = root / "docs" / "benchmarks" / "dots-tts-quant-gate-2026-07-30.md"
    if not card.is_file():
        raise FileNotFoundError(f"Missing dots.tts model card: {card}")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing dots.tts benchmark evidence: {report_path}")
    report = report_path.read_text(encoding="utf-8")
    if "**Verdict: PASS**" not in report:
        raise ValueError("dots.tts benchmark evidence does not record a PASS verdict")

    artifacts: list[dict[str, Any]] = []
    for variant, artifact_class, remote_path in DOTS_TTS_ARTIFACTS:
        local_path = model_root / remote_path
        if not local_path.is_dir():
            raise FileNotFoundError(f"Missing dots.tts release artifact: {local_path}")
        _require_exact_artifact_layout(local_path)

        metadata = json.loads(
            (local_path / "mlx_config.json").read_text(encoding="utf-8")
        )
        if metadata.get("model_family") != "dots_tts":
            raise ValueError(
                f"Invalid model family in {local_path / 'mlx_config.json'}"
            )
        if metadata.get("variant") != variant:
            raise ValueError(f"Variant mismatch in {local_path / 'mlx_config.json'}")
        if metadata.get("artifact_class") != artifact_class:
            raise ValueError(
                f"Artifact class mismatch in {local_path / 'mlx_config.json'}"
            )
        source = metadata.get("source")
        if (
            not isinstance(source, dict)
            or source.get("revision") != (DOTS_TTS_SOURCE_REVISIONS[variant])
        ):
            raise ValueError(
                f"Source revision mismatch in {local_path / 'mlx_config.json'}"
            )
        quantization = metadata.get("quantization")
        if (artifact_class == "base" and quantization is not None) or (
            artifact_class == "int8" and not isinstance(quantization, dict)
        ):
            raise ValueError(
                f"Quantization metadata mismatch in {local_path / 'mlx_config.json'}"
            )

        digest = _dots_tts_artifact_digest(local_path)
        evidence = f"`{variant}/{artifact_class}` artifact digest: `{digest}`"
        if evidence not in report:
            raise ValueError(
                f"Benchmark evidence does not match {variant}/{artifact_class} "
                f"artifact digest {digest}"
            )
        artifacts.append(
            {
                "artifact_class": artifact_class,
                "bytes": sum(
                    file.stat().st_size
                    for file in local_path.rglob("*")
                    if file.is_file()
                ),
                "digest": digest,
                "local_path": str(local_path.relative_to(root)),
                "remote_path": remote_path,
                "source_revision": source["revision"],
                "variant": variant,
            }
        )

    return {
        "artifacts": artifacts,
        "benchmark_evidence": str(report_path.relative_to(root)),
        "card": {
            "local_path": str(card.relative_to(root)),
            "remote_path": "README.md",
        },
        "include_patterns": list(DOTS_TTS_INCLUDE_PATTERNS),
        "local_root": str(model_root.relative_to(root)),
        "repo_id": DOTS_TTS_REPO_ID,
    }


def _dots_tts_card_manifest(root: Path) -> dict[str, Any]:
    card = (
        root
        / "scripts"
        / "hugging_face"
        / "model_cards"
        / "appautomaton"
        / "dots-tts-mlx.md"
    )
    if not card.is_file():
        raise FileNotFoundError(f"Missing dots.tts model card: {card}")
    content = card.read_text(encoding="utf-8")
    required = ("mlx-speech>=0.5.0", "generate_stream", "waveform streaming")
    missing = [text for text in required if text not in content]
    if missing:
        raise ValueError(f"dots.tts model card is missing release content: {missing}")
    if "runtime is inference-only and non-streaming" in content:
        raise ValueError("dots.tts model card still claims the runtime is non-streaming")
    return {
        "card": {
            "local_path": str(card.relative_to(root)),
            "remote_path": "README.md",
        },
        "repo_id": DOTS_TTS_REPO_ID,
    }


def _dots_tts_upload_commands(
    manifest: dict[str, Any], *, root: Path, hf: str
) -> tuple[list[str], list[str]]:
    model_root = root / str(manifest["local_root"])
    card = root / str(manifest["card"]["local_path"])
    artifact_command = [
        hf,
        "upload-large-folder",
        "--repo-type",
        "model",
        "--num-workers",
        "1",
        str(manifest["repo_id"]),
        str(model_root),
    ]
    for pattern in manifest["include_patterns"]:
        artifact_command.extend(("--include", str(pattern)))
    card_command = [
        hf,
        "upload",
        "--repo-type",
        "model",
        str(manifest["repo_id"]),
        str(card),
        "README.md",
    ]
    return artifact_command, card_command


def _dots_tts_card_upload_command(
    manifest: dict[str, Any], *, root: Path, hf: str
) -> list[str]:
    return [
        hf,
        "upload",
        "--repo-type",
        "model",
        str(manifest["repo_id"]),
        str(root / str(manifest["card"]["local_path"])),
        str(manifest["card"]["remote_path"]),
    ]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"Error: command exited with {result.returncode}")
        sys.exit(result.returncode)


def upload(alias: str, *, root: Path, hf: str) -> None:
    repo_id, local_rel, large = MODELS[alias]
    local_path = root / local_rel

    if not local_path.exists():
        print(f"Missing: {local_path}")
        sys.exit(1)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"

    if large:
        _run(
            [
                hf,
                "upload-large-folder",
                "--repo-type",
                "model",
                "--num-workers",
                "1",
                repo_id,
                str(local_path),
            ],
            env=env,
        )
    else:
        _run(
            [
                hf,
                "upload",
                "--repo-type",
                "model",
                repo_id,
                str(local_path),
                "mlx-int8",
            ],
            env=env,
        )

    card = (
        root
        / "scripts"
        / "hugging_face"
        / "model_cards"
        / "appautomaton"
        / f"{repo_id.partition('/')[2]}.md"
    )
    if card.is_file():
        _run(
            [hf, "upload", "--repo-type", "model", repo_id, str(card), "README.md"],
            env=env,
        )

    print(f"Done. https://huggingface.co/{repo_id}\n")


def upload_dots_tts(
    *,
    root: Path,
    hf: str | None,
    dry_run: bool,
    card_only: bool = False,
) -> None:
    manifest = (
        _dots_tts_card_manifest(root)
        if card_only
        else _dots_tts_release_manifest(root)
    )
    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if hf is None:
        raise ValueError("hf command is required for a dots.tts upload")

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    if card_only:
        _run(_dots_tts_card_upload_command(manifest, root=root, hf=hf), env=env)
    else:
        for command in _dots_tts_upload_commands(manifest, root=root, hf=hf):
            _run(command, env=env)
    print(f"Done. https://huggingface.co/{DOTS_TTS_REPO_ID}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload appautomaton MLX model artifacts to Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="MODEL",
        help=(
            "Model alias(es) to upload. Choices: "
            f"{', '.join(sorted((*MODELS, *SPECIAL_TARGETS)))}"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Upload all models.")
    parser.add_argument(
        "--list", action="store_true", help="List available models and exit."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print a release manifest without uploading.",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="For dots-tts, publish only the root README without scanning weights.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for alias, (repo_id, local, large) in sorted(MODELS.items()):
            mode = "large-folder" if large else "upload"
            print(f"  {alias:<30} → {repo_id}  [{mode}]")
        print(f"  {'dots-tts':<30} → {DOTS_TTS_REPO_ID}  [four-artifact large-folder]")
        return

    targets = [*MODELS, *SPECIAL_TARGETS] if args.all else args.targets
    if not targets:
        parser.print_help()
        sys.exit(1)

    available = {*MODELS, *SPECIAL_TARGETS}
    unknown = [t for t in targets if t not in available]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(sorted(available))}")
        sys.exit(1)
    if args.dry_run and any(target != "dots-tts" for target in targets):
        parser.error("--dry-run is currently supported only for dots-tts")
    if args.card_only and (args.all or targets != ["dots-tts"]):
        parser.error("--card-only requires the single target dots-tts")

    root = Path(__file__).resolve().parents[2]
    hf = None if args.dry_run else _resolve_hf()

    for alias in targets:
        print(f"--- {alias} ---")
        if alias == "dots-tts":
            upload_dots_tts(
                root=root,
                hf=hf,
                dry_run=args.dry_run,
                card_only=args.card_only,
            )
        else:
            if hf is None:
                raise AssertionError("hf command was not resolved")
            upload(alias, root=root, hf=hf)


if __name__ == "__main__":
    main()
