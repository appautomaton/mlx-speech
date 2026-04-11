from __future__ import annotations

import argparse
from pathlib import Path

from mlx_speech.audio.io import load_audio, write_wav
from mlx_speech.generation.longcat_audiodit import (
    parse_longcat_batch_manifest_line,
    synthesize_longcat_audiodit,
)
from mlx_speech.models.longcat_audiodit.checkpoint import (
    load_longcat_model,
    resolve_longcat_tokenizer_dir,
)
from mlx_speech.models.longcat_audiodit.tokenizer import LongCatTokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch LongCat AudioDiT generation")
    parser.add_argument("--lst", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--guidance-method", choices=["cfg", "apg"], default="cfg")
    parser.add_argument("--guidance-strength", type=float, default=4.0)
    return parser


def _run_batch_items(
    items,
    *,
    manifest_path: Path,
    output_dir: Path,
    model,
    tokenizer,
    nfe: int,
    guidance_method: str,
    guidance_strength: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        prompt_wav_path = (manifest_path.parent / item.prompt_wav_path).resolve()
        prompt_audio, _ = load_audio(
            prompt_wav_path,
            sample_rate=model.config.sampling_rate,
            mono=True,
        )
        result = synthesize_longcat_audiodit(
            model=model,
            tokenizer=tokenizer,
            text=item.gen_text,
            prompt_text=item.prompt_text,
            prompt_audio=prompt_audio[None, None, :],
            nfe=nfe,
            guidance_method=guidance_method,
            guidance_strength=guidance_strength,
            batch_mode=True,
        )
        write_wav(
            output_dir / f"{item.uid}.wav",
            result.waveform,
            sample_rate=result.sample_rate,
        )


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = Path(args.lst)
    output_dir = Path(args.output_dir)
    loaded = load_longcat_model(args.model_dir)
    tokenizer = LongCatTokenizer.from_path(
        resolve_longcat_tokenizer_dir(args.tokenizer_dir)
    )
    items = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            items.append(
                parse_longcat_batch_manifest_line(raw_line, line_number=line_number)
            )
    _run_batch_items(
        items,
        manifest_path=manifest_path,
        output_dir=output_dir,
        model=loaded.model,
        tokenizer=tokenizer,
        nfe=args.nfe,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
    )


if __name__ == "__main__":
    main()
