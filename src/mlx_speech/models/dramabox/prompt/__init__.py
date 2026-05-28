"""DramaBox prompt pipeline.

Maps `text → a_ctx [B, 1024, 2048]` via:

1. Tokenize with `LTXVGemmaTokenizer` (plain text, left-pad, no chat template)
2. Run Gemma 3 12B IT forward returning all 49 hidden states
3. `FeatureExtractorV2`: stack → per-token RMSNorm → reshape → rescale →
   audio_aggregate_embed
4. Convert binary mask to additive log-space mask
5. `Embeddings1DConnector`: replace padded slots with tiled learnable
   registers, run 8 self-attention blocks, final RMSNorm
6. Output `a_ctx [B, 1024, 2048]`
"""

from __future__ import annotations

from .feature_extractor import FeatureExtractorV2
from .additive_mask import convert_to_additive_mask
from .connector import Embeddings1DConnector
from .processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from .prompt_encoder import DramaBoxPromptEncoder, EncodedPrompt
from .checkpoint import (
    load_audio_components_state,
    load_feature_extractor_weights,
    load_connector_weights,
)

__all__ = [
    "FeatureExtractorV2",
    "convert_to_additive_mask",
    "Embeddings1DConnector",
    "EmbeddingsProcessor",
    "EmbeddingsProcessorOutput",
    "DramaBoxPromptEncoder",
    "EncodedPrompt",
    "load_audio_components_state",
    "load_feature_extractor_weights",
    "load_connector_weights",
]
