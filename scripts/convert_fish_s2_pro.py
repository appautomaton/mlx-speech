#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx

from mlx_speech.models.fish_s2_pro.codec_weights import convert_codec_pth_to_assets


DEFAULT_INPUT = Path("models/fish_s2_pro/original")


def _runtime_model_dir(input_dir: Path, output_dir: Path | None) -> Path:
    return output_dir or input_dir


def _default_codec_dir(input_dir: Path, output_dir: Path | None) -> Path:
    return _runtime_model_dir(input_dir, output_dir).parent / "codec-mlx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or repack Fish Audio S2 Pro checkpoints for local MLX use."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for repacked MLX model shards.",
    )
    parser.add_argument("--bits", type=int, default=16, choices=[8, 16])
    parser.add_argument(
        "--codec-dir",
        type=Path,
        default=None,
        help="Optional destination directory for converted codec assets.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fishaudio/s2-pro into --input-dir before repacking.",
    )
    return parser.parse_args()


def download_model(output_dir: Path) -> None:
    for cmd in (["hf", "download"], ["huggingface-cli", "download"]):
        try:
            result = subprocess.run(
                [*cmd, "fishaudio/s2-pro", "--local-dir", str(output_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return
    raise RuntimeError("Unable to download fishaudio/s2-pro")


def repack_bf16(input_dir: Path, output_dir: Path) -> None:
    shards = sorted(input_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No .safetensors shards found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        weights = mx.load(str(shard))
        mx.eval(list(weights.values()))
        mx.save_safetensors(
            str(output_dir / shard.name),
            weights,
            metadata={"format": "mlx"},
        )


def copy_supporting_files(
    input_dir: Path,
    output_dir: Path,
    *,
    codec_dir: Path | None,
) -> None:
    del codec_dir
    for path in sorted(input_dir.iterdir()):
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name != "codec.pth"
        ):
            shutil.copy2(path, output_dir / path.name)


def convert_codec_assets(codec_pth: Path, output_dir: Path) -> None:
    resolved_codec_pth = (
        codec_pth if codec_pth.suffix == ".pth" else codec_pth / "codec.pth"
    )
    convert_codec_pth_to_assets(resolved_codec_pth, output_dir)


def convert_fish_s2_pro(
    input_dir: Path,
    output_dir: Path | None,
    *,
    bits: int,
    codec_dir: Path | None = None,
) -> bool:
    if bits == 8:
        raise NotImplementedError(
            "Fish S2 Pro int8 conversion stays blocked until the faithful model tree and quantization path are real."
        )
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    if output_dir is not None:
        repack_bf16(input_dir, output_dir)
    codec_pth = input_dir / "codec.pth"
    converted_codec = False
    if output_dir is None and not codec_pth.is_file():
        raise FileNotFoundError(
            f"Missing codec.pth in {input_dir}. Codec-only conversion needs the upstream codec archive."
        )
    if codec_pth.is_file():
        convert_codec_assets(
            codec_pth, codec_dir or _default_codec_dir(input_dir, output_dir)
        )
        converted_codec = True
    if output_dir is not None:
        copy_supporting_files(input_dir, output_dir, codec_dir=codec_dir)
    return converted_codec


def main() -> None:
    args = parse_args()
    if args.download:
        download_model(args.input_dir)
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Missing input directory: {args.input_dir}. Use --download or run `hf download fishaudio/s2-pro --local-dir {args.input_dir}` first."
        )
    converted_codec = convert_fish_s2_pro(
        args.input_dir,
        args.output_dir,
        bits=args.bits,
        codec_dir=args.codec_dir,
    )
    runtime_model_dir = _runtime_model_dir(args.input_dir, args.output_dir)
    codec_output_dir = args.codec_dir or _default_codec_dir(
        args.input_dir, args.output_dir
    )
    generate_command = [
        "python scripts/generate_fish_s2_pro.py",
        f"--model-dir {runtime_model_dir}",
    ]
    if converted_codec:
        generate_command.append(f"--codec-dir {codec_output_dir}")
    if args.output_dir is None and converted_codec:
        print(f"Converted Fish S2 Pro codec assets into {codec_output_dir}")
    elif args.output_dir is not None:
        print(f"Repacked Fish S2 Pro checkpoint into {args.output_dir}")
    if converted_codec:
        print(
            "Use `"
            + " ".join(generate_command)
            + "` to synthesize from the local MLX assets."
        )


if __name__ == "__main__":
    main()
