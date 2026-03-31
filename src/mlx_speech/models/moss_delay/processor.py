"""Processor helpers for MossTTSDelay."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from ..moss_common.processor import (
    AUDIO_PLACEHOLDER,
    AssistantMessage,
    Message,
    ProcessorOutput,
    UserMessage,
    detect_text_language,
    estimate_duration_tokens,
)
from ..moss_local.processor import MossTTSLocalProcessor
from .config import MossTTSDelayConfig
from .tokenizer import MossTTSDelayTokenizer

if TYPE_CHECKING:
    from ..moss_audio_tokenizer import MossAudioTokenizerModel


class MossTTSDelayProcessor(MossTTSLocalProcessor):
    """MLX-side processor for MossTTSDelay-family inference."""

    def __init__(
        self,
        tokenizer: MossTTSDelayTokenizer,
        model_config: MossTTSDelayConfig,
        *,
        audio_tokenizer: MossAudioTokenizerModel | None = None,
    ):
        super().__init__(tokenizer=tokenizer, model_config=model_config, audio_tokenizer=audio_tokenizer)

    @classmethod
    def from_path(
        cls,
        model_dir: str | Path,
        *,
        audio_tokenizer: MossAudioTokenizerModel | None = None,
    ) -> "MossTTSDelayProcessor":
        resolved_dir = Path(model_dir)
        return cls(
            tokenizer=MossTTSDelayTokenizer.from_path(resolved_dir),
            model_config=MossTTSDelayConfig.from_path(resolved_dir),
            audio_tokenizer=audio_tokenizer,
        )

    @staticmethod
    def apply_delay_pattern(codes: mx.array, pad_code: int) -> mx.array:
        if codes.ndim != 2:
            raise ValueError(f"Expected audio codes with shape (frames, n_vq), got {codes.shape}.")
        frames = int(codes.shape[0])
        channels = int(codes.shape[1])
        delayed = mx.full((frames + channels - 1, channels), pad_code, dtype=codes.dtype)
        for channel_idx in range(channels):
            delayed[channel_idx : channel_idx + frames, channel_idx] = codes[:, channel_idx]
        return delayed

    @staticmethod
    def apply_de_delay_pattern(delay_codes: mx.array) -> mx.array:
        if delay_codes.ndim != 2:
            raise ValueError(
                f"Expected delayed audio codes with shape (frames + n_vq - 1, n_vq), got {delay_codes.shape}."
            )
        delayed_frames = int(delay_codes.shape[0])
        channels = int(delay_codes.shape[1])
        frames = delayed_frames - channels + 1
        if frames <= 0:
            raise ValueError(
                "Delay-coded tensor is too short to recover non-delayed audio codes."
            )
        codes = mx.zeros((frames, channels), dtype=delay_codes.dtype)
        for channel_idx in range(channels):
            codes[:, channel_idx] = delay_codes[channel_idx : channel_idx + frames, channel_idx]
        return codes

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: list[int],
        n_vq: int,
        gen_slot_token: str,
        delay_slot_token: str,
        audio_start_token: str,
        audio_end_token: str,
    ) -> str:
        if n_vq < 1:
            raise ValueError(f"n_vq must be >= 1, got {n_vq}")

        num_placeholders = content.count(AUDIO_PLACEHOLDER)
        if num_placeholders != len(lengths):
            raise ValueError(
                f"Number of {AUDIO_PLACEHOLDER} ({num_placeholders}) "
                f"does not match lengths ({len(lengths)})."
            )

        def build_audio_block(length: int) -> str:
            if length < 0:
                raise ValueError(f"length must be >= 0, got {length}")
            if length == 0:
                return f"{audio_start_token}{audio_end_token}"
            step_tokens = gen_slot_token * length + (delay_slot_token * (n_vq - 1))
            return f"{audio_start_token}{step_tokens}{audio_end_token}"

        lengths_iter = iter(lengths)
        return re.sub(
            re.escape(AUDIO_PLACEHOLDER),
            lambda _: build_audio_block(next(lengths_iter)),
            content,
        )

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[mx.array],
        truncation: bool = False,
    ) -> mx.array:
        if role == "user":
            audio_gen_slot_token = self.audio_user_slot_token
            audio_delay_slot_token = self.audio_user_slot_token
            truncation = False
        else:
            audio_gen_slot_token = self.audio_assistant_gen_slot_token
            audio_delay_slot_token = self.audio_assistant_delay_slot_token

        n_vq = int(audio_codes_list[0].shape[1]) if audio_codes_list else self.model_config.n_vq
        if audio_codes_list and len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
            content, audio_codes_list = self._merge_consecutive_audio_placeholders(
                content,
                audio_codes_list,
            )

        content = self._replace_audio_placeholders(
            content=content,
            lengths=[int(audio_codes.shape[0]) for audio_codes in audio_codes_list],
            n_vq=n_vq,
            gen_slot_token=audio_gen_slot_token,
            delay_slot_token=audio_delay_slot_token,
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
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
            delay_audio_codes = mx.full(
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
                delayed_audio_tokens = self.apply_delay_pattern(
                    audio_tokens.astype(mx.int32),
                    self.model_config.audio_pad_code,
                )
                pad_codes = mx.full(
                    (audio_start_idx - prefix_idx + 1, n_vq),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                pieces.extend([pad_codes, delayed_audio_tokens])
                prefix_idx = audio_end_idx

            if truncation:
                pieces[-1] = pieces[-1][: -(n_vq - 1), :]
            else:
                tail = mx.full(
                    (int(text_codes.shape[0]) - prefix_idx, n_vq),
                    self.model_config.audio_pad_code,
                    dtype=mx.int32,
                )
                pieces.append(tail)

            delay_audio_codes = mx.concatenate(pieces, axis=0)

        if text_codes.shape[0] != delay_audio_codes.shape[0]:
            text_codes = text_codes[: delay_audio_codes.shape[0]]
        return mx.concatenate([text_codes[:, None], delay_audio_codes], axis=1)

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
            input_ids_list.append(merged)

        return self._pad(input_ids_list)

    def _parse_audio_codes(self, start_length: int, audio_codes: mx.array) -> list[mx.array]:
        audio_codes = self.apply_de_delay_pattern(audio_codes)
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


__all__ = [
    "AUDIO_PLACEHOLDER",
    "AssistantMessage",
    "Message",
    "MossTTSDelayProcessor",
    "ProcessorOutput",
    "UserMessage",
    "detect_text_language",
    "estimate_duration_tokens",
]
