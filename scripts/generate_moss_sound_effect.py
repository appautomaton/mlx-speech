#!/usr/bin/env python3
"""Run local MOSS-SoundEffect inference with MLX and save a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import textwrap

import mlx.core as mx

from mlx_voice.audio import write_wav
from mlx_voice.generation import (
    MossTTSDelayGenerationConfig,
    synthesize_moss_tts_delay_conversations,
)
from mlx_voice.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_voice.models.moss_delay import (
    MossTTSDelayProcessor,
    load_moss_sound_effect_model,
    resolve_moss_sound_effect_model_dir,
)
from mlx_voice.models.moss_delay.sound_effect import (
    SOUND_EFFECT_DEFAULT_AUDIO_REPETITION_PENALTY,
    SOUND_EFFECT_DEFAULT_AUDIO_TEMPERATURE,
    SOUND_EFFECT_DEFAULT_AUDIO_TOP_K,
    SOUND_EFFECT_DEFAULT_AUDIO_TOP_P,
    SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS,
    build_sound_effect_conversation,
)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults while preserving example formatting."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MOSS-SoundEffect inference with local MLX checkpoints.\n\n"
            "This uses the shared MossTTSDelay architecture with the "
            "SoundEffect-specific checkpoint and upstream-style ambient-sound prompting."
        ),
        epilog=textwrap.dedent(
            """\
            Prompt shape:
              - ambient sound description only
              - no reference audio
              - expected token budget is estimated from duration at 12.5 tokens / second

            Example:
              python scripts/generate_moss_sound_effect.py \
                --ambient-sound "a sports car roaring past on the highway." \
                --duration-seconds 10 \
                --output outputs/sports_car.wav

            Model loading:
              - Default local runtime: models/openmoss/moss_sound_effect/mlx-4bit
              - Default local codec: models/openmoss/moss_audio_tokenizer/mlx-int8
              - Use --model-dir / --codec-dir to point at any other explicit checkpoint path
            """
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--ambient-sound",
        required=True,
        help="Natural-language description of the target sound effect or ambience.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="Target duration used to estimate expected tokens at the upstream 12.5 tokens/s heuristic.",
    )
    parser.add_argument(
        "--expected-tokens",
        type=int,
        default=None,
        help="Optional explicit expected-token override. If omitted, estimated from duration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output WAV path.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional explicit MOSS-SoundEffect checkpoint directory. Defaults to the local quantized mlx-4bit artifact.",
    )
    parser.add_argument(
        "--codec-dir",
        default=None,
        help="Optional explicit codec checkpoint directory. Defaults to the local quantized mlx-int8 artifact.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="MLX device for generation. `auto` resolves to GPU; keep CPU for explicit debugging/parity work only.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=SOUND_EFFECT_DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated row budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SOUND_EFFECT_DEFAULT_AUDIO_TEMPERATURE,
        help="Audio sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=SOUND_EFFECT_DEFAULT_AUDIO_TOP_P,
        help="Audio top-p sampling filter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=SOUND_EFFECT_DEFAULT_AUDIO_TOP_K,
        help="Audio top-k sampling filter.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=SOUND_EFFECT_DEFAULT_AUDIO_REPETITION_PENALTY,
        help="Audio repetition penalty.",
    )
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache and force the uncached generation path.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _resolve_device_name(device: str) -> str:
    if device == "auto":
        return "gpu"
    return device


def _build_generation_config(args: argparse.Namespace) -> MossTTSDelayGenerationConfig:
    return MossTTSDelayGenerationConfig(
        use_kv_cache=not args.no_kv_cache,
        max_new_tokens=args.max_new_tokens,
        audio_temperature=0.0 if args.greedy else args.temperature,
        audio_top_p=args.top_p,
        audio_top_k=args.top_k,
        audio_repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
    )


def _write_output(output_path: Path, *, waveform: mx.array, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, waveform, sample_rate=sample_rate)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    resolved_model_dir = resolve_moss_sound_effect_model_dir(args.model_dir)
    resolved_device = _resolve_device_name(args.device)
    if resolved_device == "cpu":
        mx.set_default_device(mx.cpu)
    elif resolved_device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"Unsupported device selection: {resolved_device}")

    loaded_model = load_moss_sound_effect_model(
        resolved_model_dir,
    )
    loaded_codec = load_moss_audio_tokenizer_model(
        args.codec_dir,
    )
    processor = MossTTSDelayProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )
    conversation, expected_tokens = build_sound_effect_conversation(
        processor,
        ambient_sound=args.ambient_sound,
        duration_seconds=args.duration_seconds,
        expected_tokens=args.expected_tokens,
    )
    config = _build_generation_config(args)

    result = synthesize_moss_tts_delay_conversations(
        loaded_model.model,
        processor,
        conversations=conversation,
        mode="generation",
        config=config,
    )

    output = result.outputs[0]
    output_path = Path(args.output)
    _write_output(output_path, waveform=output.waveform, sample_rate=output.sample_rate)
    elapsed_sec = time.perf_counter() - start_time

    print("Generated MOSS-SoundEffect sample")
    print(f"  output: {output_path}")
    print(f"  device: {mx.default_device()}")
    print(f"  model_dir: {resolved_model_dir}")
    print(f"  quantized: {'yes' if loaded_model.quantization is not None else 'no'}")
    print(f"  kv_cache: {'on' if config.use_kv_cache else 'off'}")
    print(f"  expected_tokens: {expected_tokens}")
    print(f"  sample_rate: {output.sample_rate}")
    print(f"  generated_rows: {int(result.generation.generated_rows.shape[1])}")
    print(f"  stop_reached: {result.generation.stop_reached}")
    print(f"  elapsed_sec: {elapsed_sec:.2f}")


if __name__ == "__main__":
    main()
