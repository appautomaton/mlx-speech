"""Granite Speech ASR model family for mlx-speech."""

from .checkpoint import (
    AlignmentReport,
    GraniteSpeechCheckpoint,
    build_alignment_report,
    load_checkpoint_into_model,
    load_granite_speech_checkpoint,
    sanitize_state_dict,
    validate_checkpoint_against_model,
)
from .config import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
    GraniteSpeechTextConfig,
)
from .encoder import (
    BatchNorm1d,
    ConformerAttention,
    ConformerBlock,
    ConformerConvModule,
    ConformerFeedForward,
    GraniteSpeechEncoder,
)
from .feature_extraction import GraniteSpeechAudioShape, GraniteSpeechFeatureExtractor
from .language_model import (
    GraniteCausalLM,
    GraniteCausalLMOutput,
    GraniteKVCache,
    GraniteLayerKVCache,
    greedy_next_token,
)
from .model import GraniteSpeechModel, GraniteSpeechModelBundle, load_granite_speech_model
from .processor import mask_audio_token_ids, replace_audio_embeddings
from .projector import (
    GraniteSpeechProjector,
    QFormerAttention,
    QFormerEncoder,
    QFormerLayer,
    QFormerModel,
)
from .tokenizer import GraniteSpeechTokenizer

__all__ = [
    "GraniteSpeechConfig",
    "GraniteSpeechEncoderConfig",
    "GraniteSpeechProjectorConfig",
    "GraniteSpeechTextConfig",
    "GraniteSpeechAudioShape",
    "GraniteSpeechCheckpoint",
    "GraniteSpeechFeatureExtractor",
    "GraniteCausalLM",
    "GraniteCausalLMOutput",
    "GraniteKVCache",
    "GraniteLayerKVCache",
    "GraniteSpeechModel",
    "GraniteSpeechModelBundle",
    "GraniteSpeechProjector",
    "GraniteSpeechTokenizer",
    "QFormerAttention",
    "QFormerEncoder",
    "QFormerLayer",
    "QFormerModel",
    "BatchNorm1d",
    "ConformerAttention",
    "ConformerBlock",
    "ConformerConvModule",
    "ConformerFeedForward",
    "GraniteSpeechEncoder",
    "AlignmentReport",
    "build_alignment_report",
    "greedy_next_token",
    "load_checkpoint_into_model",
    "load_granite_speech_checkpoint",
    "load_granite_speech_model",
    "mask_audio_token_ids",
    "replace_audio_embeddings",
    "sanitize_state_dict",
    "validate_checkpoint_against_model",
]
