#!/usr/bin/env python3
"""Run local MOSS-TTSD inference with MLX for single samples or JSONL batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import textwrap

import mlx.core as mx

from mlx_speech.audio import write_wav
from mlx_speech.generation import (
    MossTTSDelayGenerationConfig,
    synthesize_moss_tts_delay_conversations,
)
from mlx_speech.generation.moss_delay import TTSD_DEFAULT_MAX_NEW_TOKENS
from mlx_speech.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_speech.models.moss_delay import (
    MossTTSDelayProcessor,
    build_ttsd_conversation,
    load_moss_tts_delay_model,
    prepare_ttsd_sample,
    resolve_moss_tts_delay_model_dir,
    resolve_ttsd_processor_mode,
    streaming_jsonl_reader,
)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults while preserving example formatting."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MOSS-TTSD inference with local MLX checkpoints.\n\n"
            "Single-sample mode writes one WAV. Batch mode reads a JSONL file "
            "sequentially and writes `output.jsonl` plus per-sample WAV files.\n"
            "The default runtime prefers the local `mlx-int8` artifact and keeps "
            "TTSD KV cache enabled."
        ),
        epilog=textwrap.dedent(
            """\
            IMPORTANT: --text must include [S1] / [S2] speaker tags (e.g. "[S1] Hello.").
            Omitting them produces degraded or incoherent output.

            Mode requirements:
              generation: --text only
              continuation: --text plus at least one paired prompt_audio_speakerN and prompt_text_speakerN
              voice_clone: --text plus at least one prompt_audio_speakerN
              voice_clone_and_continuation: --text plus at least one paired prompt_audio_speakerN and prompt_text_speakerN

            Batch JSONL fields:
              text
              prompt_audio_speakerN
              prompt_text_speakerN

            Examples:
              python scripts/generate/moss_ttsd.py --mode generation --text "[S1] Watson, we should go now." --output outputs/ttsd.wav
              python scripts/generate/moss_ttsd.py --mode voice_clone --text "[S1] I tell you what." --prompt-audio-speaker1 outputs/source/hank_hill_ref.wav --output outputs/hank.wav
              python scripts/generate/moss_ttsd.py --mode continuation --input-jsonl inputs/ttsd.jsonl --save-dir outputs/ttsd_batch

            Model loading:
              - Default local runtime: models/openmoss/moss_ttsd/mlx-int8
              - Default local codec: models/openmoss/moss_audio_tokenizer/mlx-int8
              - Use --model-dir / --codec-dir to point at any other explicit checkpoint path
            """
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("generation", "continuation", "voice_clone", "voice_clone_and_continuation"),
        default="generation",
        help=(
            "TTSD inference mode. `generation` uses text only; `continuation` "
            "and `voice_clone_and_continuation` need paired speaker prompt "
            "audio+text; `voice_clone` uses reference audio prompts."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Dialogue text for single-sample mode, including speaker tags like [S1] and [S2].",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path for single-sample mode.",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Batch input JSONL path. Each record can include `text`, `prompt_audio_speakerN`, and `prompt_text_speakerN`.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory for batch mode. The script writes `output.jsonl` plus per-sample WAV files.",
    )
    for speaker_idx in range(1, 6):
        parser.add_argument(
            f"--prompt-audio-speaker{speaker_idx}",
            default=None,
            help=(
                f"Reference or continuation prompt audio for speaker {speaker_idx}, "
                "depending on mode."
            ),
        )
        parser.add_argument(
            f"--prompt-text-speaker{speaker_idx}",
            default=None,
            help=f"Prompt transcript for speaker {speaker_idx}. Used by continuation-style modes.",
        )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional explicit TTSD checkpoint directory. Defaults to the local quantized mlx-int8 runtime artifact.",
    )
    parser.add_argument(
        "--codec-dir",
        default=None,
        help="Optional explicit codec checkpoint directory. Defaults to the local quantized mlx-int8 runtime artifact.",
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
        default=TTSD_DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated row budget. The script reports the actual emitted row count as `generated_rows`.",
    )
    parser.add_argument("--temperature", type=float, default=1.1, help="Audio sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Audio top-p sampling filter.")
    parser.add_argument("--top-k", type=int, default=50, help="Audio top-k sampling filter.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Audio repetition penalty.",
    )
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    parser.add_argument(
        "--text-normalize",
        action="store_true",
        help="Normalize TTSD dialogue text close to upstream generation_utils.py.",
    )
    parser.add_argument(
        "--sample-rate-normalize",
        action="store_true",
        help="Normalize prompt audios through a shared minimum sample rate before encoding references.",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable TTSD KV cache and force the uncached generation path.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _resolve_device_name(device: str) -> str:
    if device == "auto":
        return "gpu"
    return device


def _resolve_io_mode(args: argparse.Namespace) -> str:
    if args.input_jsonl:
        return "batch"
    return "single"


def _collect_single_mode_speaker_maps(args: argparse.Namespace) -> tuple[dict[int, str], dict[int, str]]:
    audio_map: dict[int, str] = {}
    text_map: dict[int, str] = {}
    for speaker_idx in range(1, 6):
        audio_value = getattr(args, f"prompt_audio_speaker{speaker_idx}")
        text_value = getattr(args, f"prompt_text_speaker{speaker_idx}")
        if audio_value:
            audio_map[speaker_idx] = audio_value
        if text_value:
            text_map[speaker_idx] = text_value
    return audio_map, text_map


def _validate_args(args: argparse.Namespace) -> None:
    io_mode = _resolve_io_mode(args)
    if io_mode == "single":
        if not args.text:
            raise ValueError("Single-sample mode requires `--text`.")
        if not args.output:
            raise ValueError("Single-sample mode requires `--output`.")
        if args.save_dir:
            raise ValueError("Single-sample mode does not use `--save-dir`.")
    else:
        if not args.save_dir:
            raise ValueError("Batch mode requires `--save-dir`.")
        if args.output:
            raise ValueError("Batch mode does not use `--output`.")
        if args.text:
            raise ValueError("Batch mode does not use `--text`.")

    if io_mode != "single":
        return

    audio_map, text_map = _collect_single_mode_speaker_maps(args)
    paired_speaker_ids = sorted(set(audio_map) & set(text_map))
    if args.mode == "voice_clone" and not audio_map:
        raise ValueError("mode=voice_clone requires at least one `--prompt-audio-speakerN`.")
    if args.mode in {"continuation", "voice_clone_and_continuation"} and not paired_speaker_ids:
        raise ValueError(
            f"mode={args.mode} requires at least one paired `--prompt-audio-speakerN` and `--prompt-text-speakerN`."
        )


def _collect_single_sample_fields(args: argparse.Namespace) -> dict[str, str]:
    sample: dict[str, str] = {}
    if args.text is not None:
        sample["text"] = args.text
    for speaker_idx in range(1, 6):
        audio_value = getattr(args, f"prompt_audio_speaker{speaker_idx}")
        text_value = getattr(args, f"prompt_text_speaker{speaker_idx}")
        if audio_value is not None:
            sample[f"prompt_audio_speaker{speaker_idx}"] = audio_value
        if text_value is not None:
            sample[f"prompt_text_speaker{speaker_idx}"] = text_value
    return sample


def _build_single_sample_conversation(
    args: argparse.Namespace,
    processor: MossTTSDelayProcessor,
    *,
    n_vq: int,
) -> list[list[dict[str, object]]]:
    sample = _collect_single_sample_fields(args)
    return [[
        *build_ttsd_conversation(
            processor=processor,
            mode=args.mode,
            text=str(sample["text"]),
            audio_map={idx: sample[f"prompt_audio_speaker{idx}"] for idx in range(1, 6) if f"prompt_audio_speaker{idx}" in sample},
            text_map={idx: sample[f"prompt_text_speaker{idx}"] for idx in range(1, 6) if f"prompt_text_speaker{idx}" in sample},
            text_normalize_enabled=args.text_normalize,
            sample_rate_normalize_enabled=args.sample_rate_normalize,
            n_vq=n_vq,
        )
    ]]


def _write_jsonl_line(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_single_output(output_path: Path, *, waveform: mx.array, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, waveform, sample_rate=sample_rate)


def _run_batch(
    *,
    args: argparse.Namespace,
    model,
    processor: MossTTSDelayProcessor,
    config: MossTTSDelayGenerationConfig,
) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = save_dir / "output.jsonl"
    output_jsonl.write_text("", encoding="utf-8")

    processor_mode = resolve_ttsd_processor_mode(args.mode)
    records = streaming_jsonl_reader(args.input_jsonl)
    for line_no, raw_sample in records:
        sample_id, output_record, conversation = prepare_ttsd_sample(
            line_no=line_no,
            raw_sample=raw_sample,
            mode=args.mode,
            processor=processor,
            text_normalize_enabled=args.text_normalize,
            sample_rate_normalize_enabled=args.sample_rate_normalize,
            n_vq=processor.model_config.n_vq,
        )
        result = synthesize_moss_tts_delay_conversations(
            model,
            processor,
            conversations=[conversation],
            mode=processor_mode,
            config=config,
        )
        output = result.outputs[0]
        if int(output.waveform.shape[0]) == 0:
            output_record["output_audio"] = None
            output_record["duration"] = 0.0
        else:
            audio_path = (save_dir / f"{sample_id}.wav").resolve()
            _write_single_output(audio_path, waveform=output.waveform, sample_rate=output.sample_rate)
            output_record["output_audio"] = str(audio_path)
            output_record["duration"] = float(output.waveform.shape[0] / output.sample_rate)
        _write_jsonl_line(output_jsonl, output_record)


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


def main() -> None:
    args = parse_args()
    _validate_args(args)
    start_time = time.perf_counter()

    resolved_model_dir = resolve_moss_tts_delay_model_dir(
        args.model_dir,
    )
    resolved_device = _resolve_device_name(args.device)
    if resolved_device == "cpu":
        mx.set_default_device(mx.cpu)
    elif resolved_device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"Unsupported device selection: {resolved_device}")

    loaded_model = load_moss_tts_delay_model(
        resolved_model_dir,
    )
    loaded_codec = load_moss_audio_tokenizer_model(
        args.codec_dir,
    )
    processor = MossTTSDelayProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )
    config = _build_generation_config(args)
    processor_mode = resolve_ttsd_processor_mode(args.mode)

    if _resolve_io_mode(args) == "batch":
        _run_batch(
            args=args,
            model=loaded_model.model,
            processor=processor,
            config=config,
        )
        elapsed_sec = time.perf_counter() - start_time
        print("Generated MOSS-TTSD batch")
        print(f"  save_dir: {Path(args.save_dir)}")
        print(f"  device: {mx.default_device()}")
        print(f"  model_dir: {resolved_model_dir}")
        print(f"  quantized: {'yes' if loaded_model.quantization is not None else 'no'}")
        print(f"  kv_cache: {'on' if config.use_kv_cache else 'off'}")
        print(f"  elapsed_sec: {elapsed_sec:.2f}")
        return

    conversation = _build_single_sample_conversation(
        args,
        processor,
        n_vq=loaded_model.config.n_vq,
    )
    result = synthesize_moss_tts_delay_conversations(
        loaded_model.model,
        processor,
        conversations=conversation,
        mode=processor_mode,
        config=config,
    )

    output = result.outputs[0]
    output_path = Path(args.output)
    _write_single_output(output_path, waveform=output.waveform, sample_rate=output.sample_rate)
    elapsed_sec = time.perf_counter() - start_time

    print("Generated MOSS-TTSD sample")
    print(f"  output: {output_path}")
    print(f"  device: {mx.default_device()}")
    print(f"  model_dir: {resolved_model_dir}")
    print(f"  quantized: {'yes' if loaded_model.quantization is not None else 'no'}")
    print(f"  kv_cache: {'on' if config.use_kv_cache else 'off'}")
    print(f"  sample_rate: {output.sample_rate}")
    print(f"  generated_rows: {int(result.generation.generated_rows.shape[1])}")
    print(f"  stop_reached: {result.generation.stop_reached}")
    print(f"  elapsed_sec: {elapsed_sec:.2f}")


if __name__ == "__main__":
    main()
