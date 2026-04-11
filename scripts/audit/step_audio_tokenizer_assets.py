#!/usr/bin/env python3
"""Inspect Step-Audio tokenizer assets and prompt packing."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_speech.models.step_audio_editx import StepAudioEditXTokenizer, format_audio_token_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local Step-Audio tokenizer assets and prompt formatting.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/stepfun/step_audio_editx/original"),
        help="Step-Audio-EditX model directory containing tokenizer.json.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("models/stepfun/step_audio_tokenizer/original"),
        help="Step-Audio tokenizer asset directory.",
    )
    return parser.parse_args()


def _print_asset(path: Path, label: str) -> None:
    exists = path.exists()
    suffix = f" ({path.stat().st_size} bytes)" if exists and path.is_file() else ""
    print(f"{label}: {path} -> {'present' if exists else 'missing'}{suffix}")


def main() -> None:
    args = parse_args()

    print("Step-Audio tokenizer asset audit")
    _print_asset(args.model_dir / "tokenizer.json", "text_tokenizer_json")
    _print_asset(args.model_dir / "tokenizer.model", "text_tokenizer_model")
    _print_asset(args.model_dir / "tokenizer_config.json", "text_tokenizer_config")
    _print_asset(args.tokenizer_dir / "linguistic_tokenizer.npy", "vq02_kmeans")
    _print_asset(args.tokenizer_dir / "speech_tokenizer_v1.onnx", "vq06_model")

    funasr_root = args.tokenizer_dir / "dengcunqin"
    if funasr_root.exists():
        print(f"funasr_roots: {[p.name for p in sorted(funasr_root.iterdir())]}")
    else:
        print(f"funasr_roots: missing ({funasr_root})")

    tokenizer = StepAudioEditXTokenizer.from_path(args.model_dir)
    sample_audio_tokens = format_audio_token_string(
        [1, 2, 3, 4],
        [10, 11, 12, 13, 14, 15],
    )
    sample_clone = tokenizer.build_clone_prompt_ids(
        speaker="debug",
        prompt_text="Reference line.",
        prompt_wav_tokens=sample_audio_tokens,
        target_text="Target line.",
    )
    sample_edit = tokenizer.build_edit_prompt_ids(
        instruct_prefix="Make it calmer.",
        audio_token_str=sample_audio_tokens,
    )

    print(f"chat_template_present: {bool(tokenizer.chat_template)}")
    print(f"bos/eos/pad: {tokenizer.bos_token_id}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}")
    print(f"sample_audio_token_string: {sample_audio_tokens}")
    print(f"sample_clone_prompt_len: {len(sample_clone)}")
    print(f"sample_edit_prompt_len: {len(sample_edit)}")


if __name__ == "__main__":
    main()
