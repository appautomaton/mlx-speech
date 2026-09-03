#!/usr/bin/env python3
"""Run the pinned official FireRedTTS3 Base checkpoint on PyTorch MPS."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--inference-cfg", type=float, default=2.0)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    return parser.parse_args()


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
    from fireredtts3.llm import fireredtts3_base as base_module
    from fireredtts3.llm.fireredtts3_base import (
        FireRedTTS3Base,
        FireRedTTS3BaseCore,
    )
    from fireredtts3.redae.redae import RedAE
    from fireredtts3.utils.text_tokenizer import load_text_tokenizer

    device = torch.device("mps")
    base_module.Qwen3_1_7B_ConfigDict["attn_implementation"] = "sdpa"
    backend = FireRedTTS3Base.__new__(FireRedTTS3Base)
    backend.device = device
    backend.redae = RedAE.from_pretrained(args.model_dir / "redae").to(device).eval()
    backend.tts_core = (
        FireRedTTS3BaseCore.from_pretrained(args.model_dir / "fireredtts3_base")
        .to(device)
        .eval()
    )
    backend.text_tokenizer = load_text_tokenizer(args.model_dir / "text_tokenizer")
    backend.spk_extractor = (
        CamppEmbedding(args.model_dir / "campp/campplus_voxceleb.bin")
        .to(device)
        .eval()
    )

    prompt_audio, prompt_rate = torchaudio.load(args.reference_audio)
    generated, sample_rate = backend.generate(
        language=args.language,
        prompt_text=args.reference_text,
        prompt_audio=prompt_audio,
        prompt_audio_sr=prompt_rate,
        text=args.text,
        stop_threshold=args.stop_threshold,
        n_timesteps=args.n_timesteps,
        inference_cfg=args.inference_cfg,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, generated.cpu(), sample_rate)
    print(
        f"Wrote {args.output} "
        f"(sample_rate={sample_rate}, samples={generated.shape[-1]})"
    )


if __name__ == "__main__":
    main()
