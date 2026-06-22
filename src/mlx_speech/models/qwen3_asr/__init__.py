"""Qwen3-ASR model components.

The barrel intentionally stays lightweight: heavy submodules are imported
directly by their callers, never re-exported here. See the dependency-guard test.
"""

from .config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from .feature_extraction import (
    Qwen3ASRFeatureBatch,
    Qwen3ASRFeatureExtractor,
    _get_feat_extract_output_lengths,
)
from .processor import (
    ASR_TEXT_TAG,
    LANG_PREFIX,
    SUPPORTED_LANGUAGES,
    Qwen3ASRProcessor,
    Qwen3ASRProcessorOutput,
    Qwen3ASRPrompt,
    normalize_language_name,
    parse_asr_output,
    resolve_language,
)
from .tokenizer import Qwen3ASRTokenizer

__all__ = [
    "ASR_TEXT_TAG",
    "LANG_PREFIX",
    "Qwen3ASRAudioConfig",
    "Qwen3ASRConfig",
    "Qwen3ASRFeatureBatch",
    "Qwen3ASRFeatureExtractor",
    "Qwen3ASRProcessor",
    "Qwen3ASRProcessorOutput",
    "Qwen3ASRPrompt",
    "Qwen3ASRTextConfig",
    "Qwen3ASRThinkerConfig",
    "Qwen3ASRTokenizer",
    "SUPPORTED_LANGUAGES",
    "_get_feat_extract_output_lengths",
    "normalize_language_name",
    "parse_asr_output",
    "resolve_language",
]
