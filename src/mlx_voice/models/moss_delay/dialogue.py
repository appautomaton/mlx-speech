"""TTSD dialogue preparation helpers mirroring upstream MOSS-TTSD structure."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx

from ...audio import load_audio, resample_audio
from .processor import MossTTSDelayProcessor


def normalize_ttsd_text(text: str) -> str:
    """Normalize TTSD dialogue text close to upstream generation_utils.py."""

    normalized = re.sub(r"\[(\d+)\]", r"[S\1]", text.replace("\n", " "))
    remove_chars = '【】《》（）『』「」"\'-_“”～~‘’'
    segments = re.split(r"(?=\[S\d+\])", normalized)
    processed_parts: list[dict[str, str]] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        match = re.match(r"^(\[S\d+\])\s*(.*)", segment)
        tag, content = match.groups() if match else ("", segment)
        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        for dash in ("——", "……", "...", "⸺", "―", "—", "…"):
            content = content.replace(dash, "，")
        content = content.translate(str.maketrans({"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}))
        content = content.strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

        if len(content) > 1:
            last_char = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            content = content[:-1].replace("。", "，") + last_char

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    merged_lines: list[str] = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
    return "".join(merged_lines).replace("‘", "'").replace("’", "'")


def streaming_jsonl_reader(
    jsonl_path: str | Path,
    *,
    skip_invalid_json: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                if skip_invalid_json:
                    continue
                raise ValueError(
                    f"jsonl line {line_no}: invalid json ({exc.msg})"
                ) from exc
    return records


def _resolve_path(maybe_path: str, base_path: str | None) -> str:
    if base_path is None:
        return str(Path(maybe_path).expanduser().resolve())
    path = Path(maybe_path)
    if path.is_absolute():
        return str(path)
    return str((Path(base_path) / path).expanduser().resolve())


def collect_speaker_fields(source: dict[str, Any]) -> tuple[dict[int, str], dict[int, str], list[int]]:
    """Collect paired prompt speaker fields from args/records."""

    base_path = source.get("base_path")
    audio_map: dict[int, str] = {}
    text_map: dict[int, str] = {}
    for key, value in source.items():
        if value is None:
            continue
        key_text = str(key)
        value_text = str(value).strip()
        if not value_text:
            continue

        match_audio = re.fullmatch(r"prompt_audio_speaker(\d+)", key_text)
        if match_audio:
            speaker_id = int(match_audio.group(1))
            audio_map[speaker_id] = _resolve_path(value_text, base_path)
            continue

        match_text = re.fullmatch(r"prompt_text_speaker(\d+)", key_text)
        if match_text:
            speaker_id = int(match_text.group(1))
            text_map[speaker_id] = value_text

    speaker_ids = sorted(set(audio_map) & set(text_map))
    return audio_map, text_map, speaker_ids


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text

    merged_parts: list[str] = []
    current_tag: str | None = None
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        match = re.match(r"^(\[S\d+\])\s*(.*)", segment, re.DOTALL)
        if match:
            tag, content = match.groups()
            if tag == current_tag:
                merged_parts.append(content)
            else:
                current_tag = tag
                merged_parts.append(f"{tag}{content}")
        else:
            merged_parts.append(segment)
    return "".join(merged_parts)


def build_prefixed_ttsd_text(
    text: str,
    text_map: dict[int, str],
    speaker_ids: list[int],
) -> str:
    parts: list[str] = []
    for speaker_id in speaker_ids:
        current_text = text_map[speaker_id]
        tag = f"[S{speaker_id}]"
        if not current_text.lstrip().startswith(tag):
            current_text = f"{tag}{current_text}"
        parts.append(current_text)
    return _merge_consecutive_speaker_tags("".join(parts) + text)


def _preprocess_prompt_wavs(
    audio_map: dict[int, str],
    speaker_ids: list[int],
    *,
    target_sr: int,
    sample_rate_normalize_enabled: bool,
) -> list[mx.array]:
    loaded_wavs: list[tuple[mx.array, int]] = [
        load_audio(audio_map[speaker_id], mono=True) for speaker_id in speaker_ids
    ]
    min_sr = min(sample_rate for _, sample_rate in loaded_wavs) if sample_rate_normalize_enabled else None

    processed: list[mx.array] = []
    for wav, sample_rate in loaded_wavs:
        waveform = wav
        current_sr = int(sample_rate)
        if min_sr is not None and current_sr != int(min_sr):
            waveform = resample_audio(
                waveform,
                orig_sample_rate=current_sr,
                target_sample_rate=int(min_sr),
            )
            current_sr = int(min_sr)
        if current_sr != int(target_sr):
            waveform = resample_audio(
                waveform,
                orig_sample_rate=current_sr,
                target_sample_rate=int(target_sr),
            )
        processed.append(waveform.astype(mx.float32))
    return processed


def encode_concat_prompt_audio(
    processor: MossTTSDelayProcessor,
    audio_map: dict[int, str],
    speaker_ids: list[int],
    *,
    target_sr: int,
    sample_rate_normalize_enabled: bool,
    n_vq: int,
) -> mx.array:
    wav_list = _preprocess_prompt_wavs(
        audio_map,
        speaker_ids,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
    )
    merged = mx.concatenate(wav_list, axis=-1)
    return processor.encode_audios_from_wav([merged], sampling_rate=target_sr, n_vq=n_vq)[0]


def encode_references(
    processor: MossTTSDelayProcessor,
    audio_map: dict[int, str],
    speaker_ids: list[int],
    *,
    target_sr: int,
    sample_rate_normalize_enabled: bool,
    n_vq: int,
) -> list[mx.array | None]:
    wav_list = _preprocess_prompt_wavs(
        audio_map,
        speaker_ids,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
    )
    encoded_list = processor.encode_audios_from_wav(
        wav_list,
        sampling_rate=target_sr,
        n_vq=n_vq,
    )
    encoded_map = {speaker_id: codes for speaker_id, codes in zip(speaker_ids, encoded_list)}
    max_speaker_id = max(speaker_ids)
    return [encoded_map.get(speaker_id) for speaker_id in range(1, max_speaker_id + 1)]


def resolve_ttsd_processor_mode(mode: str) -> str:
    return "continuation" if mode in {"continuation", "voice_clone_and_continuation"} else "generation"


def build_ttsd_conversation(
    *,
    processor: MossTTSDelayProcessor,
    mode: str,
    text: str,
    audio_map: dict[int, str] | None = None,
    text_map: dict[int, str] | None = None,
    text_normalize_enabled: bool = False,
    sample_rate_normalize_enabled: bool = False,
    n_vq: int | None = None,
) -> list[dict[str, Any]]:
    if mode not in {
        "generation",
        "continuation",
        "voice_clone",
        "voice_clone_and_continuation",
    }:
        raise ValueError(f"Unsupported TTSD mode: {mode}")

    target_sr = int(processor.model_config.sampling_rate)
    n_vq = processor.model_config.n_vq if n_vq is None else int(n_vq)
    audio_map = {} if audio_map is None else dict(audio_map)
    text_map = {} if text_map is None else dict(text_map)
    if mode == "voice_clone":
        speaker_ids = sorted(audio_map)
    else:
        speaker_ids = sorted(set(audio_map) & set(text_map))

    normalized_text = normalize_ttsd_text(text) if text_normalize_enabled else text
    if mode == "generation":
        return [processor.build_user_message(text=normalized_text)]

    if not speaker_ids:
        if mode == "voice_clone":
            raise ValueError(f"mode={mode} requires at least one prompt_audio_speakerN.")
        raise ValueError(
            f"mode={mode} requires at least one paired prompt_audio_speakerN and prompt_text_speakerN."
        )

    if mode in {"continuation", "voice_clone_and_continuation"}:
        normalized_text = build_prefixed_ttsd_text(
            normalized_text,
            text_map=text_map,
            speaker_ids=speaker_ids,
        )
        if text_normalize_enabled:
            normalized_text = normalize_ttsd_text(normalized_text)

    if mode == "continuation":
        prompt_audio = encode_concat_prompt_audio(
            processor,
            audio_map,
            speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
            n_vq=n_vq,
        )
        return [
            processor.build_user_message(text=normalized_text),
            processor.build_assistant_message(audio_codes_list=[prompt_audio]),
        ]

    if mode == "voice_clone":
        reference = encode_references(
            processor,
            audio_map,
            speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
            n_vq=n_vq,
        )
        return [processor.build_user_message(text=normalized_text, reference=reference)]

    reference = encode_references(
        processor,
        audio_map,
        speaker_ids,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        n_vq=n_vq,
    )
    prompt_audio = encode_concat_prompt_audio(
        processor,
        audio_map,
        speaker_ids,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        n_vq=n_vq,
    )
    return [
        processor.build_user_message(text=normalized_text, reference=reference),
        processor.build_assistant_message(audio_codes_list=[prompt_audio]),
    ]


def prepare_ttsd_sample(
    *,
    line_no: int,
    raw_sample: dict[str, Any],
    mode: str,
    processor: MossTTSDelayProcessor,
    text_normalize_enabled: bool = False,
    sample_rate_normalize_enabled: bool = False,
    n_vq: int | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    sample = dict(raw_sample)
    sample_id = f"{line_no:06d}"
    if sample.get("text") is None:
        raise ValueError(f"jsonl line {line_no}: missing `text`")
    audio_map, text_map, _ = collect_speaker_fields(sample)
    conversation = build_ttsd_conversation(
        processor=processor,
        mode=mode,
        text=str(sample["text"]),
        audio_map=audio_map,
        text_map=text_map,
        text_normalize_enabled=text_normalize_enabled,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        n_vq=n_vq,
    )
    record = dict(raw_sample)
    record["id"] = sample_id
    for key, value in list(record.items()):
        if value is None:
            continue
        if isinstance(value, str) and re.fullmatch(r"prompt_audio_speaker\d+|output_audio|.*_path", str(key)):
            record[key] = _resolve_path(value, raw_sample.get("base_path"))
    return sample_id, record, conversation


__all__ = [
    "build_prefixed_ttsd_text",
    "build_ttsd_conversation",
    "collect_speaker_fields",
    "encode_concat_prompt_audio",
    "encode_references",
    "normalize_ttsd_text",
    "prepare_ttsd_sample",
    "resolve_ttsd_processor_mode",
    "streaming_jsonl_reader",
]
