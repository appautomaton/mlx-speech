"""Pure-MLX Gemma 3 text-only backbone shared across `mlx_speech` model families.

This module exposes the subset of Gemma 3 needed to compute all hidden states
of a text prompt under a standard causal mask. The forward pass returns the
list of 49 hidden states (embedding output + 48 decoder layers) — there is no
language modeling head, no caching, no autoregressive generation. The
backbone is consumed by feature-extraction pipelines (e.g. DramaBox's
``FeatureExtractorV2``), not as a generative model.
"""

from __future__ import annotations

from .config import GemmaTextConfig
from .model import Gemma3Model, Gemma3Output, gemma_rms_norm
from .tokenizer import LTXVGemmaTokenizer
from .checkpoint import load_gemma3_text_model

__all__ = [
    "GemmaTextConfig",
    "Gemma3Model",
    "Gemma3Output",
    "LTXVGemmaTokenizer",
    "gemma_rms_norm",
    "load_gemma3_text_model",
]
