import json
import os
from pathlib import Path

import pytest

from scripts.hugging_face.upload import (
    DOTS_TTS_ARTIFACTS,
    DOTS_TTS_INCLUDE_PATTERNS,
    DOTS_TTS_REQUIRED_FILES,
    DOTS_TTS_SOURCE_REVISIONS,
    DOTS_TTS_TOKENIZER_FILES,
    _dots_tts_card_manifest,
    _dots_tts_card_upload_command,
    _dots_tts_artifact_digest,
    _dots_tts_release_manifest,
    _dots_tts_upload_commands,
    _resolve_hf,
)


def _write_release_fixture(root: Path) -> None:
    card = root / "scripts/hugging_face/model_cards/appautomaton/dots-tts-mlx.md"
    card.parent.mkdir(parents=True)
    card.write_text("# dots.tts MLX\n", encoding="utf-8")

    evidence = ["**Verdict: PASS**", ""]
    for variant, artifact_class, remote_path in DOTS_TTS_ARTIFACTS:
        artifact = root / "models/dots_tts" / remote_path
        tokenizer = artifact / "tokenizer"
        tokenizer.mkdir(parents=True)
        for name in DOTS_TTS_REQUIRED_FILES - {"mlx_config.json"}:
            (artifact / name).write_bytes(f"{remote_path}/{name}".encode())
        for name in DOTS_TTS_TOKENIZER_FILES:
            (tokenizer / name).write_bytes(f"{remote_path}/{name}".encode())
        metadata = {
            "model_family": "dots_tts",
            "variant": variant,
            "artifact_class": artifact_class,
            "source": {"revision": DOTS_TTS_SOURCE_REVISIONS[variant]},
            "quantization": None if artifact_class == "base" else {"bits": 8},
        }
        (artifact / "mlx_config.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        digest = _dots_tts_artifact_digest(artifact)
        evidence.append(f"- `{variant}/{artifact_class}` artifact digest: `{digest}`")

    report = root / "docs/benchmarks/dots-tts-quant-gate-2026-07-30.md"
    report.parent.mkdir(parents=True)
    report.write_text("\n".join(evidence) + "\n", encoding="utf-8")


def _write_card_fixture(root: Path) -> Path:
    card = root / "scripts/hugging_face/model_cards/appautomaton/dots-tts-mlx.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(
        "mlx-speech>=0.5.0\ngenerate_stream\nwaveform streaming\n",
        encoding="utf-8",
    )
    return card


def test_release_manifest_contains_exact_four_artifact_paths(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    for unsafe in ("soar/original", "mf/original", "soar/mlx-bf16"):
        path = tmp_path / "models/dots_tts" / unsafe
        path.mkdir(parents=True)
        (path / "must-not-upload.safetensors").write_bytes(b"unsafe")

    manifest = _dots_tts_release_manifest(tmp_path)

    assert manifest["repo_id"] == "appautomaton/dots-tts-mlx"
    assert manifest["include_patterns"] == list(DOTS_TTS_INCLUDE_PATTERNS)
    assert [item["remote_path"] for item in manifest["artifacts"]] == [
        "soar/mlx-base",
        "soar/mlx-int8",
        "mf/mlx-base",
        "mf/mlx-int8",
    ]
    serialized = json.dumps(manifest)
    assert "original" not in serialized
    assert "mlx-bf16" not in serialized


def test_upload_command_can_only_select_approved_paths(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    manifest = _dots_tts_release_manifest(tmp_path)

    artifact_command, card_command = _dots_tts_upload_commands(
        manifest, root=tmp_path, hf="hf"
    )

    selected = [
        artifact_command[index + 1]
        for index, value in enumerate(artifact_command)
        if value == "--include"
    ]
    assert selected == list(DOTS_TTS_INCLUDE_PATTERNS)
    assert artifact_command[-2:] == ["--include", "mf/mlx-int8/**"]
    assert card_command[-1] == "README.md"
    assert all("original" not in value for value in artifact_command)
    assert all("mlx-bf16" not in value for value in artifact_command)


def test_card_only_manifest_and_command_never_select_weights(tmp_path: Path) -> None:
    card = _write_card_fixture(tmp_path)
    manifest = _dots_tts_card_manifest(tmp_path)
    command = _dots_tts_card_upload_command(manifest, root=tmp_path, hf="hf")

    assert manifest == {
        "card": {
            "local_path": "scripts/hugging_face/model_cards/appautomaton/dots-tts-mlx.md",
            "remote_path": "README.md",
        },
        "repo_id": "appautomaton/dots-tts-mlx",
    }
    assert command == [
        "hf",
        "upload",
        "--repo-type",
        "model",
        "appautomaton/dots-tts-mlx",
        str(card),
        "README.md",
    ]
    assert all("safetensors" not in item for item in command)


def test_card_only_manifest_rejects_stale_non_streaming_claim(tmp_path: Path) -> None:
    card = _write_card_fixture(tmp_path)
    card.write_text(
        card.read_text(encoding="utf-8")
        + "The runtime is inference-only and non-streaming.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="still claims"):
        _dots_tts_card_manifest(tmp_path)


def test_release_manifest_requires_benchmark_evidence(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    report = tmp_path / "docs/benchmarks/dots-tts-quant-gate-2026-07-30.md"
    report.unlink()

    with pytest.raises(FileNotFoundError, match="benchmark evidence"):
        _dots_tts_release_manifest(tmp_path)


def test_release_manifest_requires_pass_verdict(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    report = tmp_path / "docs/benchmarks/dots-tts-quant-gate-2026-07-30.md"
    report.write_text("**Verdict: FAIL**\n", encoding="utf-8")

    with pytest.raises(ValueError, match="PASS verdict"):
        _dots_tts_release_manifest(tmp_path)


def test_release_manifest_rejects_artifact_changed_after_gate(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    core = tmp_path / "models/dots_tts/soar/mlx-int8/core.safetensors"
    core.write_bytes(b"changed after benchmark")

    with pytest.raises(ValueError, match="does not match soar/int8"):
        _dots_tts_release_manifest(tmp_path)


def test_release_manifest_rejects_missing_runtime_file(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    tokenizer = tmp_path / "models/dots_tts/mf/mlx-base/tokenizer/tokenizer_config.json"
    tokenizer.unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        _dots_tts_release_manifest(tmp_path)


def test_release_manifest_rejects_unexpected_artifact_content(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)
    extra = tmp_path / "models/dots_tts/mf/mlx-int8/upstream-README.md"
    extra.write_text("must not be staged\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected dots.tts release content"):
        _dots_tts_release_manifest(tmp_path)


def test_hf_resolution_skips_entrypoint_with_missing_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken_dir = tmp_path / "broken"
    usable_dir = tmp_path / "usable"
    broken_dir.mkdir()
    usable_dir.mkdir()
    broken = broken_dir / "hf"
    usable = usable_dir / "hf"
    broken.write_text("#!/missing/python\n", encoding="utf-8")
    usable.write_text("#!/bin/sh\n", encoding="utf-8")
    broken.chmod(0o755)
    usable.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(broken_dir), str(usable_dir))))

    assert _resolve_hf() == str(usable)


def test_hf_resolution_fails_when_every_entrypoint_is_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "hf"
    broken.write_text("#!/missing/python\n", encoding="utf-8")
    broken.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="No usable hf CLI"):
        _resolve_hf()
