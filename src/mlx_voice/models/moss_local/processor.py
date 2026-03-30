"""Processor helpers for MossTTSLocal inference."""

from __future__ import annotations

from os import PathLike
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from ...audio import load_audio, loudness_normalize, mix_down_mono, resample_audio
from .config import MossTTSLocalConfig
from .tokenizer import MossTTSLocalTokenizer

if TYPE_CHECKING:
    from ..moss_audio_tokenizer import MossAudioTokenizerModel


AUDIO_PLACEHOLDER = "<|audio|>"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219

_USER_TEMPLATE = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""


def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


@dataclass
class Message:
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    text: str | None = None
    reference: list[mx.array | str | PathLike[str] | None] | None = None
    instruction: str | None = None
    tokens: int | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.reference is None:
            reference_text = "None"
            audio_codes_list: list[mx.array | str | PathLike[str]] = []
        else:
            reference_lines: list[str] = []
            audio_codes_list = []
            for speaker_idx, speaker_reference in enumerate(self.reference):
                if speaker_reference is None:
                    continue
                reference_lines.append(f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}")
                audio_codes_list.append(speaker_reference)
            reference_text = "\n".join(reference_lines) if reference_lines else "None"

        content = (
            _USER_TEMPLATE.replace("{reference}", str(reference_text))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{text}", str(self.text))
        )

        return {
            "role": "user",
            "content": content,
            "audio_codes_list": audio_codes_list,
        }


@dataclass
class AssistantMessage(Message):
    audio_codes_list: list[mx.array | str | PathLike[str]]
    content: str = AUDIO_PLACEHOLDER

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "audio_codes_list": self.audio_codes_list,
        }


@dataclass(frozen=True)
class ProcessorOutput:
    input_ids: mx.array
    attention_mask: mx.array


class MossTTSLocalProcessor:
    """MLX-side processor for MossTTSLocal inference modes."""

    def __init__(
        self,
        tokenizer: MossTTSLocalTokenizer,
        model_config: MossTTSLocalConfig,
        *,
        audio_tokenizer: MossAudioTokenizerModel | None = None,
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.audio_tokenizer = audio_tokenizer
        self.im_start_token_id = tokenizer.token_to_id("<|im_start|>")
        self.im_end_token_id = tokenizer.token_to_id("<|im_end|>")
        self.audio_user_slot_token = "<|audio_user_slot|>"
        self.audio_assistant_gen_slot_token = "<|audio_assistant_gen_slot|>"
        self.audio_assistant_delay_slot_token = "<|audio_assistant_delay_slot|>"
        self.audio_start_token = "<|audio_start|>"
        self.audio_end_token = "<|audio_end|>"

    @classmethod
    def from_path(
        cls,
        model_dir: str | Path,
        *,
        audio_tokenizer: MossAudioTokenizerModel | None = None,
    ) -> "MossTTSLocalProcessor":
        resolved_dir = Path(model_dir)
        return cls(
            tokenizer=MossTTSLocalTokenizer.from_path(resolved_dir),
            model_config=MossTTSLocalConfig.from_path(resolved_dir),
            audio_tokenizer=audio_tokenizer,
        )

    def with_audio_tokenizer(self, audio_tokenizer: MossAudioTokenizerModel) -> "MossTTSLocalProcessor":
        self.audio_tokenizer = audio_tokenizer
        return self

    @staticmethod
    def build_user_message(
        text: str | None = None,
        reference: list[mx.array | str | PathLike[str] | None] | None = None,
        instruction: str | None = None,
        tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        return UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        ).to_dict()

    @staticmethod
    def build_assistant_message(
        audio_codes_list: list[mx.array | str | PathLike[str]],
        content: str = AUDIO_PLACEHOLDER,
    ) -> dict[str, Any]:
        return AssistantMessage(audio_codes_list=audio_codes_list, content=content).to_dict()

    def _normalize_message(self, message: Message | dict[str, Any]) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Each message must be a Message or dict.")
        if "role" not in message:
            raise ValueError("Message dict must include a 'role' field.")
        if "content" in message and "audio_codes_list" in message:
            return message
        if message["role"] == "user":
            return self.build_user_message(
                text=message.get("text"),
                reference=message.get("reference"),
                instruction=message.get("instruction"),
                tokens=message.get("tokens"),
                quality=message.get("quality"),
                sound_event=message.get("sound_event"),
                ambient_sound=message.get("ambient_sound"),
                language=message.get("language"),
            )
        if message["role"] == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_codes_list", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role: {message['role']}")

    def _require_audio_tokenizer(self) -> MossAudioTokenizerModel:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set on processor.")
        return self.audio_tokenizer

    def encode_audios_from_wav(
        self,
        wav_list: mx.array | list[mx.array],
        sampling_rate: int,
        n_vq: int | None = None,
    ) -> list[mx.array]:
        audio_tokenizer = self._require_audio_tokenizer()
        if isinstance(wav_list, mx.array):
            wav_list = [wav_list]

        processed: list[mx.array] = []
        for wav in wav_list:
            waveform = wav.astype(mx.float32)
            if waveform.ndim == 2:
                waveform = mix_down_mono(waveform)
            if waveform.ndim != 1:
                raise ValueError(
                    f"Expected mono waveform with shape (samples,), got {waveform.shape}."
                )
            if int(sampling_rate) != int(self.model_config.sampling_rate):
                waveform = resample_audio(
                    waveform,
                    orig_sample_rate=int(sampling_rate),
                    target_sample_rate=int(self.model_config.sampling_rate),
                )
            processed.append(loudness_normalize(waveform))

        encoded = audio_tokenizer.batch_encode(processed, num_quantizers=n_vq)
        output: list[mx.array] = []
        for batch_index in range(int(encoded.audio_codes.shape[1])):
            length = int(encoded.audio_codes_lengths[batch_index].item())
            output.append(
                encoded.audio_codes[:, batch_index, :length].transpose(1, 0).astype(mx.int32)
            )
        return output

    def encode_audios_from_path(
        self,
        wav_path_list: str | PathLike[str] | list[str | PathLike[str]],
        n_vq: int | None = None,
    ) -> list[mx.array]:
        if isinstance(wav_path_list, (str, PathLike)):
            wav_path_list = [wav_path_list]

        wavs: list[mx.array] = []
        for wav_path in wav_path_list:
            waveform, sample_rate = load_audio(
                wav_path,
                sample_rate=self.model_config.sampling_rate,
                mono=True,
            )
            wavs.append(waveform)
            if sample_rate != self.model_config.sampling_rate:
                raise RuntimeError("Audio resampling failed to reach the target sampling rate.")

        return self.encode_audios_from_wav(
            wavs,
            sampling_rate=self.model_config.sampling_rate,
            n_vq=n_vq,
        )

    def decode_audio_codes(
        self,
        audio_tokens_list: mx.array | list[mx.array],
    ) -> list[mx.array]:
        audio_tokenizer = self._require_audio_tokenizer()
        if isinstance(audio_tokens_list, mx.array):
            audio_tokens_list = [audio_tokens_list]
        if len(audio_tokens_list) == 0:
            return []

        codes_list = []
        for codes in audio_tokens_list:
            if codes.ndim != 2:
                raise ValueError(
                    f"Expected audio codes with shape (frames, n_vq), got {codes.shape}."
                )
            codes_list.append(codes.transpose(1, 0).astype(mx.int32))

        decoded = audio_tokenizer.batch_decode(codes_list)
        wav_list: list[mx.array] = []
        for batch_index in range(int(decoded.audio.shape[0])):
            length = int(decoded.audio_lengths[batch_index].item())
            wav_list.append(decoded.audio[batch_index, 0, :length].astype(mx.float32))
        return wav_list

    @staticmethod
    def _replace_audio_placeholders(content: str, lengths: list[int], slot_token: str) -> str:
        num_placeholders = content.count(AUDIO_PLACEHOLDER)
        if num_placeholders != len(lengths):
            raise ValueError(
                f"Number of {AUDIO_PLACEHOLDER} markers ({num_placeholders}) "
                f"does not match audio lengths ({len(lengths)})."
            )

        def build_audio_block(length: int) -> str:
            if length < 0:
                raise ValueError(f"Audio placeholder length must be >= 0, got {length}.")
            if length == 0:
                return "<|audio_start|><|audio_end|>"
            return f"<|audio_start|>{slot_token * length}<|audio_end|>"

        lengths_iter = iter(lengths)
        return re.sub(
            re.escape(AUDIO_PLACEHOLDER),
            lambda _: build_audio_block(next(lengths_iter)),
            content,
        )

    @staticmethod
    def _merge_consecutive_audio_placeholders(
        content: str,
        audio_codes_list: list[mx.array],
    ) -> tuple[str, list[mx.array]]:
        matches = list(re.finditer(re.escape(AUDIO_PLACEHOLDER), content))
        if len(matches) <= 1:
            return content, audio_codes_list
        if len(matches) != len(audio_codes_list):
            raise ValueError(
                "Audio placeholders do not match the provided audio codes list."
            )

        new_audio_codes_list: list[mx.array] = []
        new_parts: list[str] = []
        last_pos = 0
        i = 0
        while i < len(matches):
            j = i
            while (
                j + 1 < len(matches)
                and content[matches[j].end() : matches[j + 1].start()].strip() == ""
            ):
                j += 1
            new_parts.append(content[last_pos : matches[i].start()])
            new_parts.append(AUDIO_PLACEHOLDER)
            last_pos = matches[j].end()
            if j == i:
                new_audio_codes_list.append(audio_codes_list[i])
            else:
                new_audio_codes_list.append(mx.concatenate(audio_codes_list[i : j + 1], axis=0))
            i = j + 1

        new_parts.append(content[last_pos:])
        return "".join(new_parts), new_audio_codes_list

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[mx.array],
        truncation: bool = False,
    ) -> mx.array:
        if role == "user":
            slot_token = self.audio_user_slot_token
            truncation = False
        else:
            slot_token = self.audio_assistant_gen_slot_token

        n_vq = int(audio_codes_list[0].shape[1]) if audio_codes_list else self.model_config.n_vq
        if audio_codes_list and len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
            content, audio_codes_list = self._merge_consecutive_audio_placeholders(
                content,
                audio_codes_list,
            )

        content = self._replace_audio_placeholders(
            content=content,
            lengths=[int(audio_codes.shape[0]) for audio_codes in audio_codes_list],
            slot_token=slot_token,
        )
        text_codes = mx.array(self.tokenizer.encode(content), dtype=mx.int32)
        text_code_list = [int(token_id) for token_id in text_codes.tolist()]
        audio_start_indices = [
            idx
            for idx, token_id in enumerate(text_code_list)
            if token_id == self.model_config.audio_start_token_id
        ]
        audio_end_indices = [
            idx
            for idx, token_id in enumerate(text_code_list)
            if token_id == self.model_config.audio_end_token_id
        ]
        if len(audio_start_indices) != len(audio_codes_list) or len(audio_end_indices) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match the provided audio codes list.")

        if not audio_codes_list:
            audio_codes = mx.full(
                (int(text_codes.shape[0]), n_vq),
                self.model_config.audio_pad_code,
                dtype=mx.int32,
            )
        else:
            pieces: list[mx.array] = []
            prefix_idx = 0
            for audio_start_idx_t, audio_end_idx_t, audio_tokens in zip(
                audio_start_indices,
                audio_end_indices,
                audio_codes_list,
            ):
                audio_start_idx = int(audio_start_idx_t)
                audio_end_idx = int(audio_end_idx_t)
                pad_codes = mx.full(
                    (audio_start_idx - prefix_idx + 1, n_vq),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                pieces.extend([pad_codes, audio_tokens.astype(mx.int32)])
                prefix_idx = audio_end_idx
            if not truncation:
                tail = mx.full(
                    (int(text_codes.shape[0]) - prefix_idx, n_vq),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                pieces.append(tail)
            audio_codes = mx.concatenate(pieces, axis=0)

        if text_codes.shape[0] != audio_codes.shape[0]:
            text_codes = text_codes[: audio_codes.shape[0]]
        return mx.concatenate([text_codes[:, None], audio_codes], axis=1)

    def _parse_text_codes(self, start_length: int, text_codes: mx.array) -> str:
        text = str(self.tokenizer.decode(text_codes.tolist()))
        prefix = str(self.tokenizer.decode(text_codes[:start_length].tolist()))
        text = text[len(prefix) :]

        audio_pattern = re.compile(
            rf"(?:{self.audio_start_token})?"
            rf"(?:{self.audio_assistant_gen_slot_token})*"
            rf"(?:{self.audio_assistant_delay_slot_token})*"
            rf"{self.audio_end_token}"
        )

        def normalize_audio_segments(value: str) -> str:
            def repl(match: re.Match[str]) -> str:
                segment = match.group(0)
                if self.audio_assistant_gen_slot_token in segment:
                    return AUDIO_PLACEHOLDER
                return ""

            return audio_pattern.sub(repl, value)

        return normalize_audio_segments(text)

    def _parse_audio_codes(self, start_length: int, audio_codes: mx.array) -> list[mx.array]:
        is_pad = mx.all(audio_codes == self.model_config.audio_pad_code, axis=1)
        non_pad_indices = [idx for idx, keep in enumerate((~is_pad).tolist()) if bool(keep)]
        if not non_pad_indices:
            return []

        segments: list[list[int]] = []
        current = [non_pad_indices[0]]
        for index in non_pad_indices[1:]:
            if index == current[-1] + 1:
                current.append(index)
            else:
                segments.append(current)
                current = [index]
        segments.append(current)

        audio_codes_list = [
            audio_codes[mx.array(segment, dtype=mx.int32)].astype(mx.int32)
            for segment in segments
        ]
        decoded_audio_list = self.decode_audio_codes(audio_codes_list)

        if start_length > 0 and audio_codes_list and decoded_audio_list:
            first_codes_length = int(audio_codes_list[0].shape[0])
            if first_codes_length > 0:
                trim_ratio = max(0.0, min(float(start_length) / float(first_codes_length), 1.0))
                if trim_ratio >= 1.0:
                    decoded_audio_list = decoded_audio_list[1:]
                elif trim_ratio > 0.0:
                    first_audio = decoded_audio_list[0]
                    trim_samples = int(first_audio.shape[-1] * trim_ratio)
                    decoded_audio_list[0] = first_audio[trim_samples:]

        return decoded_audio_list

    def decode_sequences(
        self,
        output: list[tuple[int, mx.array]],
    ) -> list[AssistantMessage | None]:
        generated_messages: list[AssistantMessage | None] = []
        for start_length, generation_ids in output:
            content = self._parse_text_codes(start_length, generation_ids[:, 0])
            audio_codes_list = self._parse_audio_codes(start_length, generation_ids[:, 1:])
            if content == "":
                message = None
            else:
                message = AssistantMessage(
                    content=content,
                    audio_codes_list=audio_codes_list,
                )
            generated_messages.append(message)
        return generated_messages

    def _pad(self, inputs: list[mx.array]) -> ProcessorOutput:
        lengths = [int(item.shape[0]) for item in inputs]
        batch_size = len(inputs)
        max_len = max(lengths)
        channels = inputs[0].shape[1]

        input_ids = mx.full(
            (batch_size, max_len, channels),
            self.model_config.audio_pad_code,
            dtype=mx.int32,
        )
        attention_mask = mx.zeros((batch_size, max_len), dtype=mx.bool_)

        for batch_idx, item in enumerate(inputs):
            item_len = int(item.shape[0])
            offset = max_len - item_len
            input_ids[batch_idx, offset:, :] = item
            input_ids[batch_idx, :offset, 0] = self.model_config.pad_token_id
            attention_mask[batch_idx, offset:] = True

        return ProcessorOutput(input_ids=input_ids, attention_mask=attention_mask)

    def __call__(
        self,
        conversations: Message | dict[str, Any] | list[Message | dict[str, Any]] | list[list[Message | dict[str, Any]]],
        *,
        mode: str = "generation",
    ) -> ProcessorOutput:
        if mode not in {"generation", "continuation"}:
            raise NotImplementedError("Processor supports generation and continuation modes only.")

        if isinstance(conversations, (Message, dict)):
            conversations = [conversations]

        truncation = mode == "continuation"
        input_ids_list: list[mx.array] = []
        for conversation in conversations:
            if isinstance(conversation, (Message, dict)):
                conversation = [conversation]

            normalized = [self._normalize_message(message) for message in conversation]
            if mode == "generation":
                if len(normalized) % 2 == 0:
                    raise ValueError("Generation mode expects an odd number of messages.")
                if normalized[-1]["role"] != "user":
                    raise ValueError("Generation mode expects the final message to be a user message.")
            else:
                if len(normalized) % 2 != 0:
                    raise ValueError("Continuation mode expects an even number of messages.")
                if normalized[-1]["role"] != "assistant":
                    raise ValueError("Continuation mode expects the final message to be an assistant message.")

            unified_codes = []
            for message_idx, message in enumerate(normalized):
                raw_audio_items = message.get("audio_codes_list", [])
                audio_codes_list: list[mx.array] = []
                if raw_audio_items:
                    encoded_items: list[mx.array | None] = [None] * len(raw_audio_items)
                    paths: list[str] = []
                    path_positions: list[int] = []
                    for item_idx, item in enumerate(raw_audio_items):
                        if isinstance(item, mx.array):
                            if item.ndim != 2:
                                raise TypeError(
                                    "Audio tensor items must already be tokenized as (frames, n_vq)."
                                )
                            encoded_items[item_idx] = item.astype(mx.int32)
                            continue
                        if isinstance(item, (str, PathLike)):
                            paths.append(str(item))
                            path_positions.append(item_idx)
                            continue
                        raise TypeError(
                            "Each audio item must be an audio-code tensor or a local path string."
                        )

                    if paths:
                        encoded_from_paths = self.encode_audios_from_path(paths)
                        if len(encoded_from_paths) != len(paths):
                            raise RuntimeError(
                                "encode_audios_from_path returned an unexpected number of items."
                            )
                        for position, codes in zip(path_positions, encoded_from_paths):
                            encoded_items[position] = codes
                    audio_codes_list = [item for item in encoded_items if item is not None]

                content = self.tokenizer.apply_chat_template(
                    [{"role": message["role"], "content": str(message["content"])}],
                    add_generation_prompt=(mode == "generation" and message_idx == len(normalized) - 1),
                    tokenize=False,
                )
                unified_codes.append(
                    self._get_unified_codes(
                        role=message["role"],
                        content=str(content),
                        audio_codes_list=audio_codes_list,
                        truncation=truncation,
                    )
                )

            merged = mx.concatenate(unified_codes, axis=0)
            if mode == "generation":
                audio_start_position = mx.full(
                    (1, self.model_config.n_vq + 1),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                audio_start_position[:, 0] = self.model_config.audio_start_token_id
                merged = mx.concatenate([merged, audio_start_position], axis=0)
            input_ids_list.append(merged)

        return self._pad(input_ids_list)
