#!/usr/bin/env python3
"""Run the pinned official FireRedTTS3 Base checkpoint on PyTorch MPS."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    text_input = parser.add_mutually_exclusive_group(required=True)
    text_input.add_argument("--text")
    text_input.add_argument("--text-file", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--inference-cfg", type=float, default=2.0)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--full-frontend", action="store_true")
    parser.add_argument("--token-max-n", type=int, default=80)
    parser.add_argument("--token-min-n", type=int, default=60)
    parser.add_argument("--merge-len", type=int, default=20)
    parser.add_argument("--cross-fade-ms", type=float, default=50.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _force_sdpa() -> None:
    from transformers import Qwen3Config

    original_init = Qwen3Config.__init__

    def mps_init(self, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        original_init(self, *args, **kwargs)

    Qwen3Config.__init__ = mps_init


def main() -> None:
    args = parse_args()
    import torch
    import torchaudio

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")
    _force_sdpa()

    from fireredtts3.campp.campp import CamppEmbedding
    from fireredtts3.core import FireRedTTS3
    from fireredtts3.llm import fireredtts3_base as base_module
    from fireredtts3.llm.fireredtts3_base import (
        FireRedTTS3Base,
        FireRedTTS3BaseCore,
    )
    from fireredtts3.redae.redae import RedAE
    from fireredtts3.utils.text_tokenizer import load_text_tokenizer

    device = torch.device("mps")
    base_module.Qwen3_1_7B_ConfigDict["attn_implementation"] = "sdpa"
    pipeline_type = FireRedTTS3 if args.full_frontend else FireRedTTS3Base
    backend = pipeline_type.__new__(pipeline_type)
    backend.device = device
    backend.redae = RedAE.from_pretrained(args.model_dir / "redae").to(device).eval()
    backend.tts_core = (
        FireRedTTS3BaseCore.from_pretrained(args.model_dir / "fireredtts3_base")
        .to(device)
        .eval()
    )
    backend.text_tokenizer = load_text_tokenizer(args.model_dir / "text_tokenizer")
    backend.spk_extractor = (
        CamppEmbedding(args.model_dir / "campp/campplus_voxceleb.bin").to(device).eval()
    )
    if args.full_frontend:
        backend._init_frontend(
            use_fasttext=False,
            use_llm_tn=False,
            use_wetext=False,
        )

    prompt_audio, prompt_rate = torchaudio.load(args.reference_audio)
    text = (
        args.text_file.read_text(encoding="utf-8").strip()
        if args.text_file is not None
        else args.text
    )
    generation_started = time.perf_counter()
    generate_kwargs = {
        "language": args.language,
        "prompt_text": args.reference_text,
        "prompt_audio": prompt_audio,
        "prompt_audio_sr": prompt_rate,
        "text": text,
        "stop_threshold": args.stop_threshold,
        "n_timesteps": args.n_timesteps,
        "inference_cfg": args.inference_cfg,
        "seed": args.seed,
    }
    if args.full_frontend:
        generate_kwargs.update(
            {
                "do_clean": True,
                "do_tn": False,
                "do_split": True,
                "token_max_n": args.token_max_n,
                "token_min_n": args.token_min_n,
                "merge_len": args.merge_len,
                "cross_fade_ms": args.cross_fade_ms,
            }
        )
    generated, sample_rate = backend.generate(**generate_kwargs)
    generation_seconds = time.perf_counter() - generation_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, generated.cpu(), sample_rate)
    print(
        f"Wrote {args.output} "
        f"(sample_rate={sample_rate}, samples={generated.shape[-1]}, "
        f"audio={generated.shape[-1] / sample_rate:.3f}s, "
        f"generation={generation_seconds:.3f}s, "
        f"rtf={generation_seconds / (generated.shape[-1] / sample_rate):.3f})"
    )


if __name__ == "__main__":
    main()
