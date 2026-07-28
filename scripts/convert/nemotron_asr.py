#!/usr/bin/env python3
"""Convert an extracted NVIDIA Nemotron 3.5 ASR NeMo checkpoint to MLX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

from mlx_speech.models.nemotron_asr.checkpoint import convert_nemo_state_dict
from mlx_speech.models.nemotron_asr.config import NemotronASRConfig

DEFAULT_INPUT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/original")
DEFAULT_OUTPUT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")


def _build_config(payload: dict) -> NemotronASRConfig:
    preprocessor = payload["preprocessor"]
    encoder = payload["encoder"]
    decoder = payload["decoder"]
    joint = payload["joint"]
    defaults = payload["model_defaults"]
    config = {
        "model_type": "nemotron_asr",
        "target": payload["target"],
        "preprocessor": {
            "sample_rate": preprocessor.get("sample_rate", 16_000),
            "features": preprocessor.get("features", 128),
            "n_fft": preprocessor.get("n_fft", 512),
            "window_size": preprocessor.get("window_size", 0.025),
            "window_stride": preprocessor.get("window_stride", 0.01),
            "window": preprocessor.get("window", "hann"),
            "preemph": preprocessor.get("preemph", 0.97),
            "dither": preprocessor.get("dither", 1e-5),
            "normalize": str(preprocessor.get("normalize", "NA")),
            "log_zero_guard_value": float(
                preprocessor.get("log_zero_guard_value", 2.0**-24)
            ),
            "pad_to": preprocessor.get("pad_to", 0),
            "pad_value": preprocessor.get("pad_value", 0.0),
        },
        "encoder": {
            "feat_in": encoder["feat_in"],
            "n_layers": encoder["n_layers"],
            "d_model": encoder["d_model"],
            "n_heads": encoder["n_heads"],
            "ff_expansion_factor": encoder["ff_expansion_factor"],
            "subsampling_factor": encoder["subsampling_factor"],
            "subsampling_conv_channels": encoder["subsampling_conv_channels"],
            "conv_kernel_size": encoder["conv_kernel_size"],
            "causal_downsampling": encoder.get("causal_downsampling", True),
            "conv_context_size": encoder.get("conv_context_size", "causal"),
            "conv_norm_type": encoder.get("conv_norm_type", "layer_norm"),
            "self_attention_model": encoder.get("self_attention_model", "rel_pos"),
            "att_context_style": encoder.get(
                "att_context_style", "chunked_limited"
            ),
            "att_context_size": encoder["att_context_size"],
            "pos_emb_max_len": encoder.get("pos_emb_max_len", 5000),
            "use_bias": encoder.get("use_bias", False),
            "xscaling": encoder.get("xscaling", False),
        },
        "prompt": {
            "num_prompts": defaults["num_prompts"],
            "prompt_hidden": defaults["enc_hidden"] * 2,
            "prompt_dictionary": defaults["prompt_dictionary"],
        },
        "decoder": {
            "pred_hidden": decoder["prednet"]["pred_hidden"],
            "pred_rnn_layers": decoder["prednet"]["pred_rnn_layers"],
            "vocab_size": decoder["vocab_size"],
            "blank_as_pad": decoder.get("blank_as_pad", True),
        },
        "joint": {
            "joint_hidden": joint["jointnet"]["joint_hidden"],
            "activation": joint["jointnet"]["activation"],
            "encoder_hidden": joint["jointnet"]["encoder_hidden"],
            "pred_hidden": joint["jointnet"]["pred_hidden"],
            "num_classes": joint["num_classes"],
        },
        "vocabulary": joint["vocabulary"],
        "default_language": "auto",
        "default_att_context_size": [56, 13],
        "max_symbols": payload["decoding"]["greedy"]["max_symbols"],
    }
    return NemotronASRConfig.from_dict(config)


def _load_nemo_config(path: Path) -> NemotronASRConfig:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - conversion environment only
        raise RuntimeError("Nemotron conversion requires PyYAML") from error
    with path.open(encoding="utf-8") as handle:
        return _build_config(yaml.safe_load(handle))


def _load_torch_state(path: Path) -> dict:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - conversion environment only
        raise RuntimeError(
            "Reading NeMo model_weights.ckpt requires conversion-only PyTorch; "
            "it is never imported by the MLX runtime"
        ) from error
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a state dictionary in {path}")
    return state


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> dict[str, int]:
    """Convert one extracted NeMo directory and return audit counts."""
    config = _load_nemo_config(input_dir / "model_config.yaml")
    source = _load_torch_state(input_dir / "model_weights.ckpt")
    weights, report = convert_nemo_state_dict(
        source,
        dtype=dtype,
        n_layers=config.encoder.n_layers,
        rnn_layers=config.decoder.pred_rnn_layers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx", "source": "nvidia-nemo"},
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "conversion_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    for name in (
        "427ad33c6285472cb01c3eb843d2309d_tokenizer.model",
        "c8ead90c911846569df4738620acfc0f_vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "LICENSE.OpenMDW-1.1",
        "README.md",
    ):
        source_file = input_dir / name
        if source_file.is_file():
            shutil.copy2(source_file, output_dir / name)

    return {
        "source_count": report.source_count,
        "destination_count": report.destination_count,
        "transformed_count": report.transformed_count,
        "vocabulary_size": len(config.vocabulary),
        "prompt_count": len(config.prompt.prompt_dictionary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    args = parser.parse_args()
    summary = convert(
        args.input_dir,
        args.output_dir,
        dtype=getattr(mx, args.dtype),
    )
    print(
        "Converted Nemotron: "
        f"{summary['source_count']} source -> "
        f"{summary['destination_count']} MLX tensors; "
        f"{summary['vocabulary_size']} vocabulary entries; "
        f"{summary['prompt_count']} prompt aliases"
    )


if __name__ == "__main__":
    main()
