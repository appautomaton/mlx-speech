#!/usr/bin/env python3
"""Convert Step-Audio-EditX to a pure MLX int8 bundle.

Converts Step1 LM, flow model, flow conditioner, HiFT vocoder, CampPlus
speaker embedding, and VQ02/VQ06 tokenizers from upstream .pt/.onnx/.npy
formats to .safetensors with int8 quantization where applicable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.checkpoints import get_stepfun_v4_layouts
from mlx_speech.models.step_audio_editx import (
    QuantizationConfig,
    Step1ForCausalLM,
    load_checkpoint_into_model,
    load_step_audio_editx_checkpoint,
    quantize_step_audio_editx_model,
    save_step_audio_editx_model,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
)
from mlx_speech.models.step_audio_editx.campplus import (
    load_step_audio_campplus_model,
)
from mlx_speech.models.step_audio_tokenizer import (
    load_step_audio_vq02_model,
    load_step_audio_vq06_model,
    resolve_step_audio_tokenizer_model_dir,
)


def _build_parser() -> argparse.ArgumentParser:
    layouts = get_stepfun_v4_layouts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(layouts.step_audio_editx.original_dir),
        help="Directory containing the original Step-Audio-EditX checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Step-Audio tokenizer asset directory (default: auto-resolve).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layouts.step_audio_editx.mlx_int8_dir),
        help="Directory to write the converted MLX bundle.",
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before writing.",
    )
    return parser


def _save_component(
    model,
    output_dir: Path,
    name: str,
    config_dict: dict,
    *,
    quantize: bool = False,
    quantization: QuantizationConfig | None = None,
) -> None:
    if quantize and quantization is not None:

        def _can_quantize(path: str, module) -> bool:
            if not (hasattr(module, "weight") and hasattr(module, "to_quantized")):
                return False
            return module.weight.shape[-1] % quantization.group_size == 0

        nn.quantize(
            model,
            group_size=quantization.group_size,
            bits=quantization.bits,
            mode=quantization.mode,
            class_predicate=_can_quantize,
        )
        config_dict["quantization"] = asdict(quantization)

    model.set_dtype(mx.bfloat16, lambda dt: mx.issubdtype(dt, mx.floating))
    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(list(weights.values()))

    safetensors_path = output_dir / f"{name}.safetensors"
    mx.save_safetensors(str(safetensors_path), weights, metadata={"format": "mlx"})

    config_path = output_dir / f"{name}-config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    size_mb = safetensors_path.stat().st_size / 1e6
    quant_label = f"int{quantization.bits}" if quantize else "bf16"
    print(f"  {name}: {size_mb:.0f} MB ({quant_label})")


def _copy_tokenizer_files(source_dir: Path, output_dir: Path) -> None:
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Step-Audio input directory not found: {input_dir}")

    tokenizer_dir = resolve_step_audio_tokenizer_model_dir(args.tokenizer_dir)

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    # 1. Step1 LM (int8)
    print("Converting Step1 LM...")
    checkpoint = load_step_audio_editx_checkpoint(input_dir)
    model = Step1ForCausalLM(checkpoint.config)
    load_checkpoint_into_model(model, checkpoint, strict=True)
    quantize_step_audio_editx_model(model, quantization)
    save_step_audio_editx_model(
        model, output_dir, config=checkpoint.config, quantization=quantization,
    )
    del model, checkpoint
    print("  step1: saved model.safetensors + config.json")

    # 2. Flow model (int8 — has nn.Linear in DiT + conformer)
    print("Converting flow model...")
    loaded_flow = load_step_audio_flow_model(input_dir)
    _save_component(
        loaded_flow.model, output_dir, "flow-model",
        asdict(loaded_flow.config),
        quantize=True, quantization=quantization,
    )
    del loaded_flow

    # 3. Flow conditioner (bf16 — too small for int8)
    print("Converting flow conditioner...")
    loaded_cond = load_step_audio_flow_conditioner(input_dir)
    _save_component(
        loaded_cond.model, output_dir, "flow-conditioner",
        asdict(loaded_cond.config),
    )
    del loaded_cond

    # 4. HiFT vocoder (bf16 — all conv, no meaningful nn.Linear)
    print("Converting HiFT vocoder...")
    loaded_hift = load_step_audio_hift_model(input_dir)
    _save_component(
        loaded_hift.model, output_dir, "hift",
        asdict(loaded_hift.config),
    )
    del loaded_hift

    # 5. CampPlus speaker embedding (bf16 — custom conv, no nn.Linear)
    print("Converting CampPlus...")
    loaded_camp = load_step_audio_campplus_model(input_dir)
    _save_component(
        loaded_camp.model, output_dir, "campplus",
        asdict(loaded_camp.config),
    )
    del loaded_camp

    # 6. VQ02 tokenizer (int8 — has nn.Linear in attention + FFN)
    print("Converting VQ02 tokenizer...")
    loaded_vq02 = load_step_audio_vq02_model(tokenizer_dir)
    _save_component(
        loaded_vq02.model, output_dir, "vq02",
        asdict(loaded_vq02.config),
        quantize=True, quantization=quantization,
    )
    del loaded_vq02

    # 7. VQ06 tokenizer (bf16 — custom StepAudioVQ06Linear, no nn.Linear)
    print("Converting VQ06 tokenizer...")
    loaded_vq06 = load_step_audio_vq06_model(tokenizer_dir)
    _save_component(
        loaded_vq06.model, output_dir, "vq06",
        asdict(loaded_vq06.config),
    )
    del loaded_vq06

    # 8. Copy Step1 tokenizer files
    _copy_tokenizer_files(input_dir, output_dir)

    print()
    print("Converted Step-Audio-EditX to pure MLX bundle")
    print(f"  input_dir: {input_dir}")
    print(f"  tokenizer_dir: {tokenizer_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  quantization: {quantization.mode} {quantization.bits}-bit group={quantization.group_size}")


if __name__ == "__main__":
    main()
