from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx

from .tokenizer import FishS2Tokenizer


@dataclass
class TextPart:
    text: str


@dataclass
class VQPart:
    codes: mx.array

    def __init__(self, codes):
        self.codes = mx.array(codes, dtype=mx.int32)


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[TextPart | VQPart] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    modality: Literal["text", "voice", "interleave"] | None = None


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def _validate_vq_part(self, part: VQPart, num_codebooks: int) -> None:
        if part.codes.ndim != 2:
            raise ValueError("VQ parts must be a 2-D array")
        if int(part.codes.shape[0]) == 0 or int(part.codes.shape[1]) == 0:
            raise ValueError("VQ parts must be non-empty")
        if int(part.codes.shape[0]) != num_codebooks:
            raise ValueError(
                f"VQ parts must have exactly {num_codebooks} codebook rows"
            )

    def encode_for_inference(
        self, tokenizer: FishS2Tokenizer, num_codebooks: int
    ) -> mx.array:
        segments: list[tuple[mx.array, mx.array | None]] = []

        for message in self.messages:
            if message.add_im_start:
                segments.append(
                    (mx.array([tokenizer.im_start_id], dtype=mx.int32), None)
                )
                segments.append(
                    (
                        mx.array(tokenizer.encode(f"{message.role}\n"), dtype=mx.int32),
                        None,
                    )
                )
                if message.modality is not None:
                    segments.append(
                        (
                            mx.array(
                                [tokenizer.modality_id(message.modality)],
                                dtype=mx.int32,
                            ),
                            None,
                        )
                    )

            for part in message.parts:
                if isinstance(part, TextPart):
                    segments.append(
                        (mx.array(tokenizer.encode(part.text), dtype=mx.int32), None)
                    )
                    continue

                self._validate_vq_part(part, num_codebooks)
                semantic = mx.array(
                    [tokenizer.semantic_id(code) for code in part.codes[0].tolist()],
                    dtype=mx.int32,
                )
                segments.append(
                    (semantic.astype(mx.int32), part.codes.astype(mx.int32))
                )

            if message.add_im_end:
                segments.append((mx.array([tokenizer.im_end_id], dtype=mx.int32), None))

        if not segments:
            raise ValueError("Conversation produced no prompt tokens")

        tokens = mx.concatenate([segment for segment, _ in segments], axis=0)
        values = mx.zeros((num_codebooks + 1, tokens.shape[0]), dtype=mx.int32)
        values[0] = tokens

        cursor = 0
        for segment, vq in segments:
            if vq is not None:
                length = int(segment.shape[0])
                values[1:, cursor : cursor + length] = vq
            cursor += int(segment.shape[0])

        return values
