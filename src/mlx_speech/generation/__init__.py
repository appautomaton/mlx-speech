"""Generation utilities for mlx-voice."""

from .cohere_asr import CohereAsrModel, CohereAsrResult
from .longcat_audiodit import (
    LongCatSynthesisOutput,
    generate_longcat_audiodit,
    synthesize_longcat_audiodit,
)
from .moss_delay import (
    MossTTSDelayBatchSynthesisOutput,
    MossTTSDelayGenerationConfig,
    MossTTSDelayGenerationOutput,
    MossTTSDelaySynthesisOutput,
    generate_moss_tts_delay,
    synthesize_moss_tts_delay_conversations,
)
from .moss_local import (
    MossTTSLocalBatchSynthesisOutput,
    MossTTSLocalGenerationConfig,
    MossTTSLocalGenerationOutput,
    MossTTSLocalSynthesisOutput,
    extract_audio_code_sequences,
    generate_moss_tts_local,
    sample_next_token,
    synthesize_moss_tts_local,
    synthesize_moss_tts_local_conversations,
)
from .step_audio_editx import StepAudioEditXModel, StepAudioEditXResult

try:
    from .fish_s2_pro import FishS2ProOutput, generate_fish_s2_pro
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise


__all__ = [
    "CohereAsrModel",
    "CohereAsrResult",
    "LongCatSynthesisOutput",
    "generate_longcat_audiodit",
    "MossTTSDelayBatchSynthesisOutput",
    "MossTTSDelayGenerationConfig",
    "MossTTSDelayGenerationOutput",
    "MossTTSDelaySynthesisOutput",
    "MossTTSLocalBatchSynthesisOutput",
    "MossTTSLocalGenerationConfig",
    "MossTTSLocalGenerationOutput",
    "MossTTSLocalSynthesisOutput",
    "extract_audio_code_sequences",
    "generate_moss_tts_delay",
    "generate_moss_tts_local",
    "sample_next_token",
    "StepAudioEditXModel",
    "StepAudioEditXResult",
    "synthesize_longcat_audiodit",
    "synthesize_moss_tts_delay_conversations",
    "synthesize_moss_tts_local",
    "synthesize_moss_tts_local_conversations",
]

if "FishS2ProOutput" in globals():
    __all__.extend(["FishS2ProOutput", "generate_fish_s2_pro"])
