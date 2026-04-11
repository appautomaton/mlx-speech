#!/usr/bin/env python3
"""Run a small local speech smoke-eval suite and write artifacts under outputs/tests.

This script is intentionally sequential. It exercises the real local inference entry
points for ASR and TTS-family models, saves generated WAVs, transcribes them with the
local Cohere ASR runtime, and writes a compact JSON/Markdown summary for inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if TYPE_CHECKING:
    from mlx_speech.generation.cohere_asr import CohereAsrModel


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _load_audio_resample(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        old_len = len(audio)
        new_len = int(round(old_len * target_sr / sr))
        audio = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            audio,
        ).astype(np.float32)
    return audio.astype(np.float32)


def _resolve_asr_model_dir(explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"ASR model directory not found: {path}")
        return path
    candidates = [
        REPO_ROOT / "models" / "cohere" / "cohere_transcribe" / "mlx-int8",
        REPO_ROOT / "models" / "cohere" / "cohere_transcribe" / "original",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a local Cohere ASR model directory.")


@dataclass
class SmokeCaseResult:
    name: str
    category: str
    ok: bool
    output_path: str | None
    transcript: str | None
    expected_text: str | None
    expected_match: bool | None
    duration_sec: float | None
    command: list[str] | None = None
    notes: str | None = None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_command(command: list[str], cwd: Path) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def _transcribe_file(
    asr: "CohereAsrModel",
    path: Path,
    *,
    language: str = "en",
    max_new_tokens: int = 512,
) -> str:
    audio = _load_audio_resample(path, target_sr=16000)
    return asr.transcribe(
        audio,
        sample_rate=16000,
        language=language,
        punctuation=True,
        max_new_tokens=max_new_tokens,
    ).text


def _expected_match(actual: str, expected: str) -> bool:
    return _normalize_text(expected) in _normalize_text(actual)


def _run_asr_reference_cases(asr: "CohereAsrModel", output_dir: Path) -> list[SmokeCaseResult]:
    cases = [
        (
            "hank_ref",
            REPO_ROOT / "outputs" / "source" / "hank_hill_ref.wav",
            "Loud is not allowed. Now you listen.",
        ),
        (
            "peggy_ref",
            REPO_ROOT / "outputs" / "source" / "peggy_hill_ref.wav",
            "All right. You're disappointed. I get it.",
        ),
        (
            "sherlock_ref",
            REPO_ROOT / "outputs" / "clone_eval" / "github_references" / "sherlock2_split_1.wav",
            "It was not that he felt any emotion akin to love for Irene Adler.",
        ),
    ]

    results: list[SmokeCaseResult] = []
    asr_dir = output_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    for name, path, expected in cases:
        transcript = _transcribe_file(asr, path)
        match = _expected_match(transcript, expected)
        _write_text(asr_dir / f"{name}.txt", transcript + "\n")
        info = sf.info(path)
        results.append(
            SmokeCaseResult(
                name=name,
                category="asr_reference",
                ok=match,
                output_path=str(path.relative_to(REPO_ROOT)),
                transcript=transcript,
                expected_text=expected,
                expected_match=match,
                duration_sec=round(info.frames / info.samplerate, 3),
            )
        )
    return results


def _run_tts_case(
    *,
    name: str,
    command: list[str],
    output_path: Path,
    expected_text: str,
    asr: "CohereAsrModel",
    output_dir: Path,
) -> SmokeCaseResult:
    exit_code, stdout, stderr, elapsed = _run_command(command, REPO_ROOT)
    log_path = output_dir / "logs" / f"{name}.log"
    _write_text(
        log_path,
        "\n".join(
            [
                "COMMAND:",
                " ".join(command),
                "",
                "STDOUT:",
                stdout,
                "",
                "STDERR:",
                stderr,
            ]
        ),
    )
    if exit_code != 0:
        return SmokeCaseResult(
            name=name,
            category="tts_generation",
            ok=False,
            output_path=str(output_path.relative_to(REPO_ROOT)),
            transcript=None,
            expected_text=expected_text,
            expected_match=False,
            duration_sec=elapsed,
            command=command,
            notes=f"Generator exited with code {exit_code}. See {log_path.relative_to(REPO_ROOT)}.",
        )

    transcript = _transcribe_file(asr, output_path)
    match = _expected_match(transcript, expected_text)
    _write_text(output_dir / "transcripts" / f"{name}.txt", transcript + "\n")
    info = sf.info(output_path)
    return SmokeCaseResult(
        name=name,
        category="tts_generation",
        ok=match,
        output_path=str(output_path.relative_to(REPO_ROOT)),
        transcript=transcript,
        expected_text=expected_text,
        expected_match=match,
        duration_sec=round(info.frames / info.samplerate, 3),
        command=command,
    )


def _build_tts_commands(output_dir: Path) -> list[tuple[str, list[str], Path, str]]:
    text = "This is a local smoke test for the MLX speech runtime."
    return [
        (
            "moss_local_generation",
            [
                sys.executable,
                "scripts/generate/moss_local.py",
                "--text",
                text,
                "--mode",
                "generation",
                "--output",
                str(output_dir / "generated" / "moss_local_smoke.wav"),
                "--greedy",
                "--max-new-tokens",
                "96",
            ],
            output_dir / "generated" / "moss_local_smoke.wav",
            text,
        ),
        (
            "moss_ttsd_voice_clone",
            [
                sys.executable,
                "scripts/generate/moss_ttsd.py",
                "--mode",
                "voice_clone",
                "--text",
                "[S1] This is a local smoke test for the MLX speech runtime.",
                "--prompt-audio-speaker1",
                "outputs/source/hank_hill_ref.wav",
                "--output",
                str(output_dir / "generated" / "moss_ttsd_smoke.wav"),
                "--greedy",
                "--max-new-tokens",
                "120",
            ],
            output_dir / "generated" / "moss_ttsd_smoke.wav",
            text,
        ),
        (
            "vibevoice_generation",
            [
                sys.executable,
                "scripts/generate/vibevoice.py",
                "--text",
                text,
                "--reference-audio",
                "outputs/source/hank_hill_ref.wav",
                "--output",
                str(output_dir / "generated" / "vibevoice_smoke.wav"),
                "--diffusion-steps",
                "10",
                "--seed",
                "123",
                "--max-new-tokens",
                "256",
            ],
            output_dir / "generated" / "vibevoice_smoke.wav",
            text,
        ),
    ]


def _render_report(results: list[SmokeCaseResult]) -> str:
    lines = ["# Local Speech Smoke Eval", ""]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"## {result.name} [{status}]")
        lines.append(f"- category: {result.category}")
        if result.output_path is not None:
            lines.append(f"- output: `{result.output_path}`")
        if result.duration_sec is not None:
            lines.append(f"- duration_sec: {result.duration_sec}")
        if result.expected_text:
            lines.append(f"- expected: {result.expected_text}")
        if result.transcript:
            lines.append(f"- transcript: {result.transcript}")
        if result.expected_match is not None:
            lines.append(f"- expected_match: {result.expected_match}")
        if result.notes:
            lines.append(f"- notes: {result.notes}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/tests",
        help="Directory for generated smoke-eval artifacts and summary files.",
    )
    parser.add_argument(
        "--asr-model-dir",
        default=None,
        help="Optional Cohere ASR model directory. Defaults to local mlx-int8, then original.",
    )
    return parser.parse_args()


def main() -> None:
    from mlx_speech.generation.cohere_asr import CohereAsrModel

    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated").mkdir(parents=True, exist_ok=True)
    (output_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    asr_model_dir = _resolve_asr_model_dir(args.asr_model_dir)
    print(f"Loading Cohere ASR from {asr_model_dir} ...")
    asr = CohereAsrModel.from_dir(asr_model_dir)

    results = _run_asr_reference_cases(asr, output_dir)

    for name, command, wav_path, expected_text in _build_tts_commands(output_dir):
        print(f"Running {name} ...")
        results.append(
            _run_tts_case(
                name=name,
                command=command,
                output_path=wav_path,
                expected_text=expected_text,
                asr=asr,
                output_dir=output_dir,
            )
        )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "asr_model_dir": str(asr_model_dir),
        "all_ok": all(result.ok for result in results),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(results), encoding="utf-8")

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}")
        if result.transcript:
            print(f"  transcript: {result.transcript}")

    print(f"\nSummary written to {output_dir / 'summary.json'}")
    print(f"Report written to {output_dir / 'report.md'}")
    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
