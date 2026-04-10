from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlx_speech.models.longcat_audiodit.checkpoint import (
    load_checkpoint_into_model,
    load_longcat_checkpoint,
    quantize_longcat_model,
    resolve_longcat_tokenizer_dir,
    save_longcat_model,
)
from mlx_speech.models.longcat_audiodit.config import QuantizationConfig
from mlx_speech.models.longcat_audiodit.model import LongCatAudioDiTModel
from mlx_speech.models.longcat_audiodit.text_encoder import LongCatUMT5Encoder
from mlx_speech.models.longcat_audiodit.transformer import LongCatAudioDiTTransformer
from mlx_speech.models.longcat_audiodit.vae import LongCatAudioDiTVae


DEFAULT_INPUT = "models/longcat_audiodit/original"
DEFAULT_OUTPUT = "models/longcat_audiodit/mlx-int8"
DEFAULT_TOKENIZER = "models/longcat_audiodit/tokenizer/umt5-base"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LongCat AudioDiT 3.5B to MLX int8"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--skip-tokenizer-copy", action="store_true")
    return parser


def _copy_tokenizer_tree(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint = load_longcat_checkpoint(args.input_dir)
    model = LongCatAudioDiTModel(
        checkpoint.config,
        text_encoder=LongCatUMT5Encoder(checkpoint.config.text_encoder_config),
        transformer=LongCatAudioDiTTransformer(checkpoint.config),
        vae=LongCatAudioDiTVae(checkpoint.config.vae_config),
    )
    load_checkpoint_into_model(model, checkpoint, strict=True)
    quantization = QuantizationConfig(
        bits=args.bits, group_size=args.group_size, mode=args.mode
    )
    quantize_longcat_model(model, quantization)
    save_longcat_model(
        model, args.output_dir, config=checkpoint.config, quantization=quantization
    )
    if not args.skip_tokenizer_copy:
        source_tokenizer = Path(args.tokenizer_dir)
        destination_tokenizer = resolve_longcat_tokenizer_dir()
        if source_tokenizer.resolve() != destination_tokenizer.resolve():
            _copy_tokenizer_tree(source_tokenizer, destination_tokenizer)


if __name__ == "__main__":
    main()
