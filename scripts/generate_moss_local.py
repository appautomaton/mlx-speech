#!/usr/bin/env python3
"""Generate speech with local MLX weights and save a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_speech.audio import normalize_peak, trim_leading_silence, write_wav
from mlx_speech.generation import (
    MossTTSLocalGenerationConfig,
    synthesize_moss_tts_local_conversations,
)
from mlx_speech.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_speech.models.moss_local import (
    MossTTSLocalProcessor,
    estimate_duration_tokens,
    load_moss_tts_local_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument(
        "--mode",
        choices=("generation", "clone", "continuation", "continue_clone"),
        default="generation",
        help="Inference mode to run.",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Local reference audio path used for clone and continuation modes.",
    )
    parser.add_argument(
        "--output",
        default="outputs/moss_local.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional MossTTSLocal checkpoint directory. Defaults to local mlx-int8.",
    )
    parser.add_argument(
        "--codec-dir",
        default=None,
        help="Optional codec checkpoint directory. Defaults to local mlx-int8.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Maximum number of generated rows.",
    )
    parser.add_argument(
        "--no-max-new-tokens",
        action="store_true",
        help="Do not set a user-visible row limit; rely on EOS with an internal safety cap.",
    )
    parser.add_argument(
        "--safety-max-new-tokens",
        type=int,
        default=2048,
        help="Internal safety cap used when --no-max-new-tokens is enabled.",
    )
    parser.add_argument(
        "--n-vq",
        type=int,
        default=None,
        help="Number of RVQ layers to generate/decode.",
    )
    parser.add_argument(
        "--expected-tokens",
        type=int,
        default=None,
        help="Optional duration-control token target for the user prompt.",
    )
    parser.add_argument(
        "--auto-estimate-expected-tokens",
        action="store_true",
        help="Enable upstream-style duration estimation when --expected-tokens is omitted.",
    )
    parser.add_argument(
        "--no-estimate-expected-tokens",
        action="store_true",
        help="Compatibility flag; duration estimation is already off by default.",
    )
    parser.add_argument("--instruction", default=None, help="Optional instruction field.")
    parser.add_argument("--quality", default=None, help="Optional quality field.")
    parser.add_argument("--sound-event", default=None, help="Optional sound event field.")
    parser.add_argument("--ambient-sound", default=None, help="Optional ambient sound field.")
    parser.add_argument("--language", default=None, help="Optional language field.")
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=1.5,
        help="Text-channel sampling temperature.",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=50,
        help="Text-channel top-k sampling filter.",
    )
    parser.add_argument(
        "--text-top-p",
        type=float,
        default=1.0,
        help="Text-channel top-p sampling filter.",
    )
    parser.add_argument(
        "--text-repetition-penalty",
        type=float,
        default=1.0,
        help="Text-channel repetition penalty.",
    )
    parser.add_argument(
        "--audio-temperature",
        type=float,
        default=1.7,
        help="Audio-channel sampling temperature. Matches the upstream Gradio app default.",
    )
    parser.add_argument(
        "--audio-top-k",
        type=int,
        default=25,
        help="Audio-channel top-k sampling filter. Matches the upstream Gradio app default.",
    )
    parser.add_argument(
        "--audio-top-p",
        type=float,
        default=0.8,
        help="Audio-channel top-p sampling filter. Matches the upstream Gradio app default.",
    )
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=1.0,
        help="Audio-channel repetition penalty. Matches the upstream Gradio app default.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    kv_group = parser.add_mutually_exclusive_group()
    kv_group.add_argument(
        "--kv-cache",
        action="store_true",
        help="Enable the default KV-cache runtime explicitly.",
    )
    kv_group.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache and fall back to the uncached loop.",
    )
    parser.add_argument(
        "--trim-leading-silence",
        action="store_true",
        help="Trim leading low-energy audio before writing the WAV.",
    )
    parser.add_argument(
        "--normalize-peak",
        type=float,
        default=0.0,
        help="Target peak amplitude for output normalization. Set <= 0 to disable.",
    )
    return parser.parse_args()


def resolve_cache_flag(args: argparse.Namespace) -> bool:
    if args.no_kv_cache:
        return False
    return True


def supports_duration_control(mode: str) -> bool:
    return mode not in {"continuation", "continue_clone"}


def build_conversation(args: argparse.Namespace, processor: MossTTSLocalProcessor) -> tuple[list[list[dict]], str]:
    resolved_tokens = args.expected_tokens
    if supports_duration_control(args.mode) and resolved_tokens is None and args.auto_estimate_expected_tokens:
        _, resolved_tokens, _, _ = estimate_duration_tokens(args.text)
    if not supports_duration_control(args.mode):
        resolved_tokens = None
    user_kwargs: dict[str, object] = {
        "text": args.text,
        "instruction": args.instruction,
        "tokens": resolved_tokens,
        "quality": args.quality,
        "sound_event": args.sound_event,
        "ambient_sound": args.ambient_sound,
        "language": args.language,
    }
    if args.mode == "generation":
        return [[processor.build_user_message(**user_kwargs)]], "generation"
    if not args.reference_audio:
        raise ValueError(f"--reference-audio is required for mode={args.mode}.")
    if args.mode == "clone":
        user_kwargs["reference"] = [args.reference_audio]
        return [[processor.build_user_message(**user_kwargs)]], "generation"
    if args.mode == "continuation":
        return [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[args.reference_audio]),
            ]
        ], "continuation"

    user_kwargs["reference"] = [args.reference_audio]
    return [
        [
            processor.build_user_message(**user_kwargs),
            processor.build_assistant_message(audio_codes_list=[args.reference_audio]),
        ]
    ], "continuation"


def main() -> None:
    args = parse_args()

    loaded_model = load_moss_tts_local_model(args.model_dir)
    loaded_codec = load_moss_audio_tokenizer_model(args.codec_dir)
    processor = MossTTSLocalProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )
    use_kv_cache = resolve_cache_flag(args)
    generation_config = MossTTSLocalGenerationConfig(
        max_new_tokens=None if args.no_max_new_tokens else args.max_new_tokens,
        safety_max_new_tokens=args.safety_max_new_tokens,
        n_vq_for_inference=args.n_vq,
        text_temperature=args.text_temperature,
        text_top_k=args.text_top_k,
        text_top_p=args.text_top_p,
        text_repetition_penalty=args.text_repetition_penalty,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
        audio_top_p=args.audio_top_p,
        audio_repetition_penalty=args.audio_repetition_penalty,
        do_sample=False if args.greedy else None,
        use_kv_cache=use_kv_cache,
    )
    conversations, processor_mode = build_conversation(args, processor)
    result = synthesize_moss_tts_local_conversations(
        loaded_model.model,
        processor,
        loaded_codec.model,
        conversations=conversations,
        mode=processor_mode,
        config=generation_config,
    )
    synthesis = result.outputs[0]
    waveform = synthesis.waveform
    if args.trim_leading_silence:
        waveform = trim_leading_silence(
            waveform,
            sample_rate=synthesis.sample_rate,
        )
    if args.normalize_peak > 0:
        waveform = normalize_peak(
            waveform,
            target_peak=args.normalize_peak,
        )

    output_path = write_wav(
        Path(args.output),
        waveform,
        sample_rate=synthesis.sample_rate,
    )
    audio_frames = (
        int(synthesis.generation.audio_codes_list[0].shape[0])
        if synthesis.generation.audio_codes_list
        else 0
    )

    print("Generated MossTTSLocal sample")
    print(f"  output: {output_path}")
    print(f"  mode: {args.mode}")
    print(f"  sample_rate: {synthesis.sample_rate}")
    print(f"  generated_rows: {int(synthesis.generation.generated_rows.shape[1])}")
    print(f"  audio_frames: {audio_frames}")
    print(f"  stop_reached: {synthesis.generation.stop_reached}")


if __name__ == "__main__":
    main()
