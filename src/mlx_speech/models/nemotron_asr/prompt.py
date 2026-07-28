"""Language-ID prompt conditioning for Nemotron 3.5 ASR."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import PromptArgs


def build_prompt_kernel(d_model: int, args: PromptArgs) -> list[nn.Module]:
    """Build the MLP while preserving checkpoint indices ``0`` and ``2``."""
    return [
        nn.Linear(d_model + args.num_prompts, args.prompt_hidden),
        nn.ReLU(),
        nn.Linear(args.prompt_hidden, d_model),
    ]


def resolve_prompt_index(language: str, args: PromptArgs) -> int:
    """Resolve one documented language alias, rejecting silent fallbacks."""
    try:
        index = args.prompt_dictionary[language]
    except KeyError as error:
        raise ValueError(f"unsupported Nemotron language prompt: {language!r}") from error
    if not 0 <= index < args.num_prompts:
        raise ValueError(f"prompt index {index} is outside [0, {args.num_prompts})")
    return index


def apply_language_prompt(
    encoded: mx.array,
    language: str,
    args: PromptArgs,
    kernel: list[nn.Module],
) -> mx.array:
    """Concatenate a broadcast one-hot prompt, then project to encoder width."""
    if encoded.ndim != 3:
        raise ValueError(f"expected encoded features [B, T, D], got {encoded.shape}")
    index = resolve_prompt_index(language, args)
    prompt = (mx.arange(args.num_prompts) == index).astype(encoded.dtype)
    prompt = mx.broadcast_to(prompt, (*encoded.shape[:2], args.num_prompts))
    output = mx.concatenate([encoded, prompt], axis=-1)
    for layer in kernel:
        output = layer(output)
    return output.astype(encoded.dtype)


__all__ = [
    "apply_language_prompt",
    "build_prompt_kernel",
    "resolve_prompt_index",
]
