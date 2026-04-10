from __future__ import annotations

import argparse

from mlx_speech.generation.longcat_audiodit import generate_longcat_audiodit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongCat AudioDiT 3.5B generation")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--output-audio", required=True)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--guidance-method", choices=["cfg", "apg"], default="cfg")
    parser.add_argument("--guidance-strength", type=float, default=4.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    generate_longcat_audiodit(
        text=args.text,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio,
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_audio=args.output_audio,
        nfe=args.nfe,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
    )


if __name__ == "__main__":
    main()
