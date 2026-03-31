#!/usr/bin/env python3
"""Benchmark TTSD generation runtime on the local MLX quantized TTSD path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import mlx.core as mx

from mlx_voice.generation import MossTTSDelayGenerationConfig, generate_moss_tts_delay
from mlx_voice.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_voice.models.moss_delay import (
    MossTTSDelayProcessor,
    load_moss_tts_delay_model,
    resolve_moss_tts_delay_model_dir,
)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    max_new_tokens: int


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: PromptSpec
    sampling_mode: str
    use_kv_cache: bool
    weights: str


@dataclass(frozen=True)
class BenchmarkResult:
    prompt: str
    sampling_mode: str
    kv_cache: bool
    weights: str
    model_dir: str
    quantized: bool
    generated_rows: int
    stop_reached: bool
    load_sec: float
    prepare_sec: float
    generation_sec: float
    decode_sec: float
    end_to_end_sec: float


PROMPT_SPECS: dict[str, PromptSpec] = {
    "short": PromptSpec(
        name="short",
        text="[S1] Watson, we should go now.",
        max_new_tokens=96,
    ),
    "medium": PromptSpec(
        name="medium",
        text=(
            "[S1] Watson, should we go now? The rain is getting worse, and the road is "
            "flooding. Can you hear the thunder?"
        ),
        max_new_tokens=160,
    ),
    "long": PromptSpec(
        name="long",
        text=(
            "[S1] Watson, the station is quieter than I expected tonight. The rain has "
            "washed the streets clean, but the wind still sounds uneasy. We should not "
            "stay here too long, because the last train may already be delayed. Did you "
            "hear that door close behind us, or was it only the draft? Stay close, and "
            "keep your voice low until we reach the platform."
        ),
        max_new_tokens=320,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="MLX device for the benchmark.",
    )
    parser.add_argument(
        "--prompts",
        default="short,medium,long",
        help="Comma-separated prompt set to benchmark: short,medium,long.",
    )
    parser.add_argument(
        "--weights",
        default="quantized",
        help=(
            "Comma-separated weight set to benchmark. The local default runtime is "
            "`quantized`; use `--model-dir` for any custom explicit checkpoint path."
        ),
    )
    parser.add_argument(
        "--cache",
        default="on,off",
        help="Comma-separated cache settings to benchmark: on,off.",
    )
    parser.add_argument(
        "--modes",
        default="greedy,sampled",
        help="Comma-separated sampling modes to benchmark: greedy,sampled.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional explicit TTSD checkpoint directory. If set, weight selection flags are ignored.",
    )
    parser.add_argument(
        "--codec-dir",
        default=None,
        help="Optional explicit codec checkpoint directory.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def _parse_csv_choices(raw: str, *, valid: set[str], argument_name: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{argument_name} must include at least one choice.")
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(
            f"{argument_name} contains invalid choices: {', '.join(invalid)}."
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    prompt_names = _parse_csv_choices(
        args.prompts,
        valid=set(PROMPT_SPECS),
        argument_name="--prompts",
    )
    weight_names = _parse_csv_choices(
        args.weights,
        valid={"quantized"},
        argument_name="--weights",
    )
    cache_names = _parse_csv_choices(
        args.cache,
        valid={"on", "off"},
        argument_name="--cache",
    )
    sampling_modes = _parse_csv_choices(
        args.modes,
        valid={"greedy", "sampled"},
        argument_name="--modes",
    )

    cases: list[BenchmarkCase] = []
    for prompt_name in prompt_names:
        prompt = PROMPT_SPECS[prompt_name]
        for sampling_mode in sampling_modes:
            for weight_name in weight_names:
                for cache_name in cache_names:
                    cases.append(
                        BenchmarkCase(
                            prompt=prompt,
                            sampling_mode=sampling_mode,
                            use_kv_cache=(cache_name == "on"),
                            weights=weight_name,
                        )
                    )
    return cases


def _build_generation_config(case: BenchmarkCase) -> MossTTSDelayGenerationConfig:
    greedy = case.sampling_mode == "greedy"
    return MossTTSDelayGenerationConfig(
        use_kv_cache=case.use_kv_cache,
        max_new_tokens=case.prompt.max_new_tokens,
        text_temperature=0.0 if greedy else 1.5,
        audio_temperature=0.0 if greedy else 1.1,
        text_top_k=50,
        text_top_p=1.0,
        audio_top_k=50,
        audio_top_p=0.9,
        audio_repetition_penalty=1.1,
        do_sample=not greedy,
    )


def _build_conversation(processor: MossTTSDelayProcessor, prompt: PromptSpec) -> list[list[dict[str, str]]]:
    return [[processor.build_user_message(text=prompt.text)]]


def _force_eval_message_audio(messages) -> None:
    audio_tensors = [
        audio
        for message in messages
        if message is not None
        for audio in message.audio_codes_list
    ]
    if audio_tensors:
        mx.eval(audio_tensors)


def _run_benchmark_case(
    case: BenchmarkCase,
    *,
    model_dir: str | None,
    codec_dir: str | None,
) -> BenchmarkResult:
    case_start = time.perf_counter()
    resolved_model_dir = resolve_moss_tts_delay_model_dir(
        model_dir,
    )

    load_start = time.perf_counter()
    loaded_model = load_moss_tts_delay_model(
        resolved_model_dir,
    )
    loaded_codec = load_moss_audio_tokenizer_model(
        codec_dir,
    )
    processor = MossTTSDelayProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )
    load_sec = time.perf_counter() - load_start

    prepare_start = time.perf_counter()
    conversations = _build_conversation(processor, case.prompt)
    batch = processor(conversations, mode="generation")
    prepare_sec = time.perf_counter() - prepare_start

    config = _build_generation_config(case)

    warm_generation = generate_moss_tts_delay(
        loaded_model.model,
        batch.input_ids,
        batch.attention_mask,
        config=config,
    )
    mx.eval(warm_generation.generated_rows)
    warm_messages = processor.decode_sequences(list(warm_generation.messages))
    _force_eval_message_audio(warm_messages)

    generation_start = time.perf_counter()
    generation = generate_moss_tts_delay(
        loaded_model.model,
        batch.input_ids,
        batch.attention_mask,
        config=config,
    )
    mx.eval(generation.generated_rows)
    generation_sec = time.perf_counter() - generation_start

    decode_start = time.perf_counter()
    decoded_messages = processor.decode_sequences(list(generation.messages))
    _force_eval_message_audio(decoded_messages)
    decode_sec = time.perf_counter() - decode_start

    return BenchmarkResult(
        prompt=case.prompt.name,
        sampling_mode=case.sampling_mode,
        kv_cache=case.use_kv_cache,
        weights="custom" if model_dir is not None else case.weights,
        model_dir=str(resolved_model_dir),
        quantized=loaded_model.quantization is not None,
        generated_rows=int(generation.generated_rows.shape[1]),
        stop_reached=generation.stop_reached,
        load_sec=load_sec,
        prepare_sec=prepare_sec,
        generation_sec=generation_sec,
        decode_sec=decode_sec,
        end_to_end_sec=time.perf_counter() - case_start,
    )


def _print_human_summary(results: list[BenchmarkResult], *, device_name: str) -> None:
    print("TTSD benchmark summary")
    print(f"  device: {device_name}")
    print(f"  runs: {len(results)}")
    for result in results:
        print(
            "  - "
            f"prompt={result.prompt} "
            f"mode={result.sampling_mode} "
            f"weights={result.weights} "
            f"cache={'on' if result.kv_cache else 'off'} "
            f"generated_rows={result.generated_rows} "
            f"stop_reached={result.stop_reached} "
            f"load_sec={result.load_sec:.2f} "
            f"generation_sec={result.generation_sec:.2f} "
            f"decode_sec={result.decode_sec:.2f} "
            f"end_to_end_sec={result.end_to_end_sec:.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    cases = _build_cases(args)
    results = [
        _run_benchmark_case(
            case,
            model_dir=args.model_dir,
            codec_dir=args.codec_dir,
        )
        for case in cases
    ]

    _print_human_summary(results, device_name=args.device)
    payload = {
        "device": args.device,
        "results": [asdict(result) for result in results],
    }
    json_summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    print(f"JSON_SUMMARY={json_summary}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
