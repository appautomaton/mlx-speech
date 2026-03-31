#!/usr/bin/env python3
"""Sweep clone presets over a fixed eval manifest and write WAV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_voice.audio import write_wav
from mlx_voice.generation import (
    MossTTSLocalGenerationConfig,
    synthesize_moss_tts_local_conversations,
)
from mlx_voice.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_voice.models.moss_local import (
    MossTTSLocalProcessor,
    estimate_duration_tokens,
    load_moss_tts_local_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="examples/clone_eval/macos_builtin_en.json",
        help="Path to the committed eval manifest.",
    )
    parser.add_argument(
        "--reference-dir",
        default="outputs/clone_eval/macos_builtin_en/references",
        help="Directory containing generated reference WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/clone_eval/macos_builtin_en/runs",
        help="Directory for generated clone outputs and summary JSON.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=("clone-v1", "app-defaults"),
        help="Preset(s) to evaluate. Defaults to both.",
    )
    parser.add_argument(
        "--duration-control",
        choices=("off", "on", "both"),
        default="both",
        help="Whether to run clone generation with explicit expected_tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional explicit generation limit. Defaults to EOS with a safety cap.",
    )
    parser.add_argument(
        "--safety-max-new-tokens",
        type=int,
        default=512,
        help="Internal safety cap when max_new_tokens is unset.",
    )
    parser.add_argument(
        "--limit-references",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Optional limit for smoke runs.",
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preset_config(name: str, *, max_new_tokens: int | None, safety_max_new_tokens: int) -> MossTTSLocalGenerationConfig:
    base_kwargs = {
        "max_new_tokens": max_new_tokens,
        "safety_max_new_tokens": safety_max_new_tokens,
        "use_kv_cache": True,
    }
    if name == "clone-v1":
        return MossTTSLocalGenerationConfig.clone_v1_defaults(**base_kwargs)
    if name == "app-defaults":
        return MossTTSLocalGenerationConfig.app_defaults(**base_kwargs)
    raise ValueError(f"Unsupported preset: {name}")


def duration_modes(value: str) -> list[bool]:
    if value == "off":
        return [False]
    if value == "on":
        return [True]
    return [False, True]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = args.preset or ["clone-v1", "app-defaults"]
    duration_flags = duration_modes(args.duration_control)

    loaded_model = load_moss_tts_local_model()
    loaded_codec = load_moss_audio_tokenizer_model()
    processor = MossTTSLocalProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )

    references = manifest["references"]
    prompts = manifest["prompts"]
    if args.limit_references is not None:
        references = references[: args.limit_references]
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    summary: list[dict] = []

    for reference in references:
        reference_path = reference_dir / f"{reference['id']}.wav"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference audio: {reference_path}")

        for prompt in prompts:
            for preset in presets:
                for use_duration_control in duration_flags:
                    expected_tokens = None
                    if use_duration_control:
                        _, expected_tokens, _, _ = estimate_duration_tokens(prompt["text"])

                    config = preset_config(
                        preset,
                        max_new_tokens=args.max_new_tokens,
                        safety_max_new_tokens=args.safety_max_new_tokens,
                    )
                    conversations = [[
                        processor.build_user_message(
                            text=prompt["text"],
                            reference=[str(reference_path)],
                            tokens=expected_tokens,
                        )
                    ]]
                    result = synthesize_moss_tts_local_conversations(
                        loaded_model.model,
                        processor,
                        loaded_codec.model,
                        conversations=conversations,
                        mode="generation",
                        config=config,
                    )
                    synthesis = result.outputs[0]
                    duration_tag = "tokens-on" if use_duration_control else "tokens-off"
                    run_dir = output_dir / preset / duration_tag / reference["id"]
                    run_dir.mkdir(parents=True, exist_ok=True)
                    output_path = run_dir / f"{prompt['id']}.wav"
                    write_wav(output_path, synthesis.waveform, sample_rate=synthesis.sample_rate)

                    summary.append(
                        {
                            "reference_id": reference["id"],
                            "prompt_id": prompt["id"],
                            "preset": preset,
                            "duration_control": use_duration_control,
                            "expected_tokens": expected_tokens,
                            "output_path": str(output_path),
                            "sample_rate": synthesis.sample_rate,
                            "waveform_samples": int(synthesis.waveform.shape[0]),
                            "waveform_seconds": float(synthesis.waveform.shape[0]) / float(synthesis.sample_rate),
                            "generated_rows": int(synthesis.generation.generated_rows.shape[1]),
                            "audio_frames": int(synthesis.generation.audio_codes_list[0].shape[0]),
                            "stop_reached": bool(synthesis.generation.stop_reached),
                        }
                    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print("Completed clone preset sweep")
    print(f"  manifest: {args.manifest}")
    print(f"  reference_dir: {reference_dir}")
    print(f"  runs: {len(summary)}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
