"""Packing helpers for Step-Audio dual tokenizer codes."""

from __future__ import annotations

import re
from typing import Iterable


AUDIO_TOKEN_RE = re.compile(r"<audio_(\d+)>")


def _truncate_to_full_groups(
    values: list[int],
    *,
    group_size: int,
    strict: bool,
) -> list[int]:
    remainder = len(values) % group_size
    if remainder == 0:
        return values
    if strict:
        raise ValueError(
            f"Expected a multiple of {group_size} Step-Audio tokens, got {len(values)}."
        )
    return values[: len(values) - remainder]


def raw_vq02_to_prompt_tokens(vq02_tokens: Iterable[int]) -> list[int]:
    return [int(token) for token in vq02_tokens]


def raw_vq06_to_prompt_tokens(
    vq06_tokens: Iterable[int],
    *,
    vq06_offset: int = 1024,
) -> list[int]:
    return [int(token) + int(vq06_offset) for token in vq06_tokens]


def prompt_vq02_to_raw_tokens(vq02_tokens: Iterable[int]) -> list[int]:
    return [int(token) for token in vq02_tokens]


def prompt_vq06_to_raw_tokens(
    vq06_tokens: Iterable[int],
    *,
    vq06_offset: int = 1024,
) -> list[int]:
    return [int(token) - int(vq06_offset) for token in vq06_tokens]


def prompt_tokens_to_mixed_ids(
    prompt_tokens: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
) -> list[int]:
    return [int(token) + int(audio_token_base_id) for token in prompt_tokens]


def mixed_ids_to_prompt_tokens(
    mixed_ids: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
) -> list[int]:
    return [int(token) - int(audio_token_base_id) for token in mixed_ids]


def raw_vq02_to_mixed_ids(
    vq02_tokens: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
) -> list[int]:
    return prompt_tokens_to_mixed_ids(
        raw_vq02_to_prompt_tokens(vq02_tokens),
        audio_token_base_id=audio_token_base_id,
    )


def raw_vq06_to_mixed_ids(
    vq06_tokens: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
    vq06_offset: int = 1024,
) -> list[int]:
    return prompt_tokens_to_mixed_ids(
        raw_vq06_to_prompt_tokens(vq06_tokens, vq06_offset=vq06_offset),
        audio_token_base_id=audio_token_base_id,
    )


def interleave_step_audio_tokens(
    vq02_tokens: Iterable[int],
    vq06_tokens: Iterable[int],
    *,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
) -> list[int]:
    """Interleave Step-Audio dual streams using the exact upstream 2+3 pattern."""

    vq02 = [int(token) for token in vq02_tokens]
    vq06 = [int(token) for token in vq06_tokens]
    groups = min(len(vq02) // interleave_vq02, len(vq06) // interleave_vq06)
    mixed: list[int] = []
    for group_idx in range(groups):
        vq02_start = group_idx * interleave_vq02
        vq06_start = group_idx * interleave_vq06
        mixed.extend(vq02[vq02_start : vq02_start + interleave_vq02])
        mixed.extend(vq06[vq06_start : vq06_start + interleave_vq06])
    return mixed


def deinterleave_step_audio_tokens(
    interleaved_tokens: Iterable[int],
    *,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
    strict: bool = False,
) -> tuple[list[int], list[int]]:
    """Split an interleaved Step-Audio sequence back into the two streams."""

    group_size = interleave_vq02 + interleave_vq06
    mixed = _truncate_to_full_groups(
        [int(token) for token in interleaved_tokens],
        group_size=group_size,
        strict=strict,
    )

    vq02: list[int] = []
    vq06: list[int] = []
    for group_start in range(0, len(mixed), group_size):
        vq02.extend(mixed[group_start : group_start + interleave_vq02])
        vq06.extend(mixed[group_start + interleave_vq02 : group_start + group_size])
    return vq02, vq06


def pack_raw_codes_to_prompt_tokens(
    vq02_tokens: Iterable[int],
    vq06_tokens: Iterable[int],
    *,
    vq06_offset: int = 1024,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
) -> list[int]:
    return interleave_step_audio_tokens(
        raw_vq02_to_prompt_tokens(vq02_tokens),
        raw_vq06_to_prompt_tokens(vq06_tokens, vq06_offset=vq06_offset),
        interleave_vq02=interleave_vq02,
        interleave_vq06=interleave_vq06,
    )


def pack_raw_codes_to_mixed_ids(
    vq02_tokens: Iterable[int],
    vq06_tokens: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
    vq06_offset: int = 1024,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
) -> list[int]:
    return interleave_step_audio_tokens(
        raw_vq02_to_mixed_ids(vq02_tokens, audio_token_base_id=audio_token_base_id),
        raw_vq06_to_mixed_ids(
            vq06_tokens,
            audio_token_base_id=audio_token_base_id,
            vq06_offset=vq06_offset,
        ),
        interleave_vq02=interleave_vq02,
        interleave_vq06=interleave_vq06,
    )


def unpack_prompt_tokens_to_raw_codes(
    prompt_tokens: Iterable[int],
    *,
    vq06_offset: int = 1024,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
    strict: bool = False,
) -> tuple[list[int], list[int]]:
    vq02_prompt, vq06_prompt = deinterleave_step_audio_tokens(
        prompt_tokens,
        interleave_vq02=interleave_vq02,
        interleave_vq06=interleave_vq06,
        strict=strict,
    )
    return (
        prompt_vq02_to_raw_tokens(vq02_prompt),
        prompt_vq06_to_raw_tokens(vq06_prompt, vq06_offset=vq06_offset),
    )


def unpack_mixed_ids_to_raw_codes(
    mixed_ids: Iterable[int],
    *,
    audio_token_base_id: int = 65536,
    vq06_offset: int = 1024,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
    strict: bool = False,
) -> tuple[list[int], list[int]]:
    return unpack_prompt_tokens_to_raw_codes(
        mixed_ids_to_prompt_tokens(mixed_ids, audio_token_base_id=audio_token_base_id),
        vq06_offset=vq06_offset,
        interleave_vq02=interleave_vq02,
        interleave_vq06=interleave_vq06,
        strict=strict,
    )


def prompt_tokens_to_audio_token_string(prompt_tokens: Iterable[int]) -> str:
    return "".join(f"<audio_{int(token)}>" for token in prompt_tokens)


def parse_audio_token_string(text: str) -> list[int]:
    return [int(match.group(1)) for match in AUDIO_TOKEN_RE.finditer(text)]


def format_audio_token_string(
    vq02_tokens: Iterable[int],
    vq06_tokens: Iterable[int],
    *,
    vq06_offset: int = 1024,
    interleave_vq02: int = 2,
    interleave_vq06: int = 3,
) -> str:
    return prompt_tokens_to_audio_token_string(
        pack_raw_codes_to_prompt_tokens(
            vq02_tokens,
            vq06_tokens,
            vq06_offset=vq06_offset,
            interleave_vq02=interleave_vq02,
            interleave_vq06=interleave_vq06,
        )
    )
