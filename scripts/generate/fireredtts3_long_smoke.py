#!/usr/bin/env python3
"""Generate the reusable FireRedTTS3 Chinese long-text smoke sample."""

from __future__ import annotations

import argparse
import math
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from mlx_speech import tts
from mlx_speech.audio import write_wav


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models/firered/firered_tts3/mlx-bf16"
DEFAULT_REFERENCE_AUDIO = Path("/tmp/fireredtts3-mlx-port/reference.wav")
DEFAULT_OUTPUT = Path("/tmp/fireredtts3-mlx-port/long-text-smoke.wav")
DEFAULT_REFERENCE_TEXT = "For Timothy was a spoiled cat, and he allowed no one."
DEFAULT_TEXT = (
    "这个项目的目标，是在苹果芯片上提供完全基于MLX的开源语音推理库，"
    "并以统一接口支持多种语音模型。FireRedTTS三代基础版的输入包括目标文本、"
    "参考音频和逐字匹配的参考文本。音频首先被重采样到二十四千赫兹。"
    "RedAE编码器把每四百八十个采样点映射为隐藏表示，再用十八层Qwen3式"
    "分组查询变换器和六十四词元滑窗注意力压缩成六十四维潜变量。"
    "CAM++说话人网络从滤波器组特征出发，依次经过卷积前端、稠密时延网络、"
    "上下文门控与统计池化，得到五百一十二维音色向量。文本分词后进入二十八层"
    "Qwen3自回归骨干，隐藏宽度为二千零四十八，使用十六个查询头和八个键值头。"
    "模型先执行一次提示预填充，随后依靠键值缓存逐块生成。PatchEncoder用八层"
    "变换器把四帧语音潜变量映射回语言模型空间。DiT由十一层扩散变换块组成，"
    "每层结合自注意力、前馈网络、卷积模块和自适应层归一化，在说话人条件、"
    "骨干条件及两块历史潜变量的共同约束下，根据余弦时间表执行十步欧拉积分，"
    "从噪声恢复下一块语音。最后，RedAE解码器重建频谱，并通过逆短时傅里叶"
    "变换输出单声道波形。当前优化把滑窗注意力限制在有界查询块内，以BF16"
    "执行局部线性层，并只编译固定十二帧的DiT张量区域；可变键值缓存、停止判断"
    "和内存清理仍保持即时执行。黄金测试中，核心生成速度提升约百分之二十，"
    "峰值显存基本不变。这个长文本测试将继续检查模型在长上下文中的稳定性、"
    "停顿、内容完整性和实时系数。"
)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LIST_PATTERN = re.compile(r"^(\s*)[-+*]\s+", flags=re.MULTILINE)
_HEADING_PATTERN = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
_EMPHASIS_PATTERN = re.compile(r"(?<!\w)(?:\*\*|__|~~)(.+?)(?:\*\*|__|~~)(?!\w)")
_PUNCTUATION = {
    "zh": frozenset("。？！；、.?!;"),
    "other": frozenset(".?!;"),
}


def _is_emoji(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0x1F3FB <= codepoint <= 0x1F3FF
    )


def clean_text(text: str) -> str:
    """Apply lightweight cleanup in this caller-owned text layer."""

    if not isinstance(text, str):
        raise TypeError("FireRedTTS3 text must be a string")
    text = bytes(text, "utf-8").decode("utf-8", "ignore")
    filtered: list[str] = []
    for character in text:
        if character in ("\u2028", "\u2029", "\u00a0"):
            filtered.append(" ")
            continue
        category = unicodedata.category(character)
        if category in {"Co", "Cs"}:
            continue
        if category == "Cf" and character != "\u200d":
            continue
        if _is_emoji(character):
            continue
        filtered.append(character)
    text = "".join(filtered).replace("\ufffd", "")
    text = _IMAGE_PATTERN.sub("", text)
    text = _LINK_PATTERN.sub(r"\1", text)
    text = _LIST_PATTERN.sub(r"\1", text)
    text = _HEADING_PATTERN.sub("", text)
    text = _EMPHASIS_PATTERN.sub(r"\1", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _is_decimal_dot(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def split_text(
    text: str,
    *,
    language: str,
    token_count: Callable[[str], int] | None = None,
    token_max_n: int = 80,
    token_min_n: int = 60,
    merge_len: int = 20,
) -> list[str]:
    """Apply the upstream soft split policy outside the model runtime."""

    if token_max_n <= 0 or token_min_n < 0 or merge_len < 0:
        raise ValueError("FireRedTTS3 split lengths must be non-negative")
    if language != "Chinese" and token_count is None:
        raise ValueError("token_count is required for non-Chinese text splitting")

    def measure(value: str) -> int:
        if language == "Chinese":
            return len(value)
        assert token_count is not None
        return int(token_count(value))

    punctuation = _PUNCTUATION["zh" if language == "Chinese" else "other"]
    utterances: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in punctuation:
            continue
        if character == "." and _is_decimal_dot(text, index):
            continue
        if start < index:
            utterance = text[start : index + 1]
            if index + 1 < len(text) and text[index + 1] in {'"', "”"}:
                utterance += text[index + 1]
                start = index + 2
            else:
                start = index + 1
            utterances.append(utterance)
    if start < len(text):
        utterances.append(text[start:])
    if not utterances:
        utterances.append(text + ("。" if language == "Chinese" else ""))

    segments: list[str] = []
    current = ""
    for utterance in utterances:
        if (
            measure(current + utterance) > token_max_n
            and measure(current) > token_min_n
        ):
            segments.append(current)
            current = ""
        current += utterance
    if current:
        if measure(current) < merge_len and segments:
            segments[-1] += current
        else:
            segments.append(current)
    return segments


def cross_fade_waveforms(
    segments: list[mx.array],
    *,
    sample_rate: int,
    cross_fade_ms: float = 50.0,
) -> mx.array:
    """Join caller-generated mono segments with a linear overlap."""

    if not segments:
        raise ValueError("FireRedTTS3 cross-fade requires at least one segment")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(cross_fade_ms) or cross_fade_ms < 0:
        raise ValueError("cross_fade_ms must be finite and non-negative")
    output = segments[0]
    if output.ndim != 1:
        raise ValueError("FireRedTTS3 waveform segments must be mono")
    requested_fade = round(cross_fade_ms / 1000.0 * sample_rate)
    for segment in segments[1:]:
        if segment.ndim != 1:
            raise ValueError("FireRedTTS3 waveform segments must be mono")
        fade_length = min(requested_fade, int(output.size), int(segment.size))
        if fade_length <= 0:
            output = mx.concatenate((output, segment))
            continue
        ramp = mx.linspace(0.0, 1.0, fade_length, dtype=mx.float32).astype(output.dtype)
        overlap = (
            output[-fade_length:] * (1.0 - ramp)
            + segment[:fade_length].astype(output.dtype) * ramp
        )
        output = mx.concatenate((output[:-fade_length], overlap, segment[fade_length:]))
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=DEFAULT_REFERENCE_AUDIO,
    )
    parser.add_argument("--reference-text", default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--segments-file",
        type=Path,
        help="UTF-8 text file containing one prepared utterance per non-empty line.",
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--max-audio-patches", type=int, default=400)
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable the official Base text splitting for a diagnostic run.",
    )
    parser.add_argument(
        "--cross-fade-ms",
        type=float,
        default=50.0,
        help="Linear overlap used when joining generated text segments.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"FireRedTTS3 artifact not found: {args.model_dir}")
    if not args.reference_audio.is_file():
        raise FileNotFoundError(
            f"FireRedTTS3 reference audio not found: {args.reference_audio}"
        )
    load_started = time.perf_counter()
    model = tts.load(str(args.model_dir))
    load_seconds = time.perf_counter() - load_started
    if args.segments_file is not None:
        if args.no_split:
            raise ValueError("--segments-file and --no-split cannot be used together")
        segments = [
            line.strip()
            for line in args.segments_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not segments:
            raise ValueError("--segments-file must contain a prepared utterance")
        source_text = "".join(segments)
    else:
        source_text = clean_text(args.text)
        segments = (
            [source_text]
            if args.no_split
            else split_text(
                source_text,
                language=args.language,
                token_count=(
                    None if args.language == "Chinese" else model.tokenizer.count_tokens
                ),
            )
        )

    mx.reset_peak_memory()
    generation_started = time.perf_counter()
    generated_segments: list[mx.array] = []
    sample_rate: int | None = None
    for segment in segments:
        result = model.generate(
            segment,
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            language=args.language,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            flow_steps=args.flow_steps,
            stop_threshold=args.stop_threshold,
            max_audio_patches=args.max_audio_patches,
        )
        if sample_rate is not None and result.sample_rate != sample_rate:
            raise RuntimeError(
                f"FireRedTTS3 sample rate changed: {sample_rate} -> {result.sample_rate}"
            )
        sample_rate = result.sample_rate
        generated_segments.append(result.waveform)
    assert sample_rate is not None
    waveform = cross_fade_waveforms(
        generated_segments,
        sample_rate=sample_rate,
        cross_fade_ms=args.cross_fade_ms,
    )
    mx.eval(waveform)
    sample_count = int(waveform.size)
    output = write_wav(
        args.output,
        waveform,
        sample_rate=sample_rate,
    )
    generation_seconds = time.perf_counter() - generation_started
    audio_seconds = sample_count / sample_rate
    chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", source_text))
    print(
        "FireRedTTS3 long smoke "
        f"output={output} chinese_chars={chinese_characters} "
        f"text_codepoints={len(source_text)} segments={len(segments)} "
        f"audio={audio_seconds:.3f}s load={load_seconds:.3f}s "
        f"generation={generation_seconds:.3f}s "
        f"rtf={generation_seconds / audio_seconds:.3f} "
        f"mlx_peak={mx.get_peak_memory() / 1024**3:.3f}GiB"
    )
    return output


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
