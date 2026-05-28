"""Top-level prompt encoder: text → ``a_ctx``.

Holds references to the Gemma 3 backbone, the tokenizer, and the
`EmbeddingsProcessor`. Exposes a single ``encode(text, max_length=1024)``
method returning the audio-context tensor that feeds the DiT cross-attention.

Reference: `.references/DramaBox/ltx2/ltx_pipelines/utils/blocks.py:440-488`
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..ltx.rope import LTXRopeType
from ...gemma3_text import Gemma3Model, LTXVGemmaTokenizer
from .processor import EmbeddingsProcessor


@dataclass
class EncodedPrompt:
    """Output of `DramaBoxPromptEncoder.encode`.

    `a_ctx` is the audio-context tensor ``[B, T, 2048]`` consumed by the
    DiT's cross-attention. `attention_mask` is the post-connector binary
    mask (all-ones; included for downstream symmetry).
    """

    a_ctx: mx.array
    attention_mask: mx.array


class DramaBoxPromptEncoder(nn.Module):
    """Text → `a_ctx`. Holds the Gemma backbone and the processor pipeline."""

    def __init__(
        self,
        *,
        gemma: Gemma3Model,
        tokenizer: LTXVGemmaTokenizer,
        processor: EmbeddingsProcessor,
    ):
        super().__init__()
        # Hold references but do NOT register Gemma as a child — it has its
        # own weight management (quantized 4-bit) and we don't want
        # `model.load_weights(...)` to try to update its parameters via
        # this object. Keep it as a non-module attribute.
        object.__setattr__(self, "_gemma", gemma)
        self.tokenizer = tokenizer
        self.processor = processor

    @property
    def gemma(self) -> Gemma3Model:
        return self._gemma  # type: ignore[attr-defined]

    def encode(
        self,
        text: str,
        *,
        max_length: int = 1024,
    ) -> EncodedPrompt:
        """Encode a single prompt string into `a_ctx`."""
        input_ids, attention_mask = self.tokenizer.encode(text, max_length=max_length)
        gemma_out = self.gemma(input_ids, attention_mask)
        processed = self.processor(gemma_out.hidden_states, attention_mask)
        return EncodedPrompt(
            a_ctx=processed.audio_encoding,
            attention_mask=processed.attention_mask,
        )


__all__ = ["DramaBoxPromptEncoder", "EncodedPrompt"]
