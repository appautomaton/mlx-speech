"""Generation utilities for mlx-voice.

Imports are lazy so each model family only loads when accessed,
avoiding cross-family dependency pollution (e.g. fish_s2_pro
requiring ``transformers`` should not block longcat_audiodit).
"""

__all__ = [
    "CohereAsrModel",
    "CohereAsrResult",
    "FishS2ProOutput",
    "LongCatSynthesisOutput",
    "generate_fish_s2_pro",
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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CohereAsrModel": (".cohere_asr", "CohereAsrModel"),
    "CohereAsrResult": (".cohere_asr", "CohereAsrResult"),
    "FishS2ProOutput": (".fish_s2_pro", "FishS2ProOutput"),
    "generate_fish_s2_pro": (".fish_s2_pro", "generate_fish_s2_pro"),
    "LongCatSynthesisOutput": (".longcat_audiodit", "LongCatSynthesisOutput"),
    "generate_longcat_audiodit": (".longcat_audiodit", "generate_longcat_audiodit"),
    "synthesize_longcat_audiodit": (".longcat_audiodit", "synthesize_longcat_audiodit"),
    "MossTTSDelayBatchSynthesisOutput": (".moss_delay", "MossTTSDelayBatchSynthesisOutput"),
    "MossTTSDelayGenerationConfig": (".moss_delay", "MossTTSDelayGenerationConfig"),
    "MossTTSDelayGenerationOutput": (".moss_delay", "MossTTSDelayGenerationOutput"),
    "MossTTSDelaySynthesisOutput": (".moss_delay", "MossTTSDelaySynthesisOutput"),
    "generate_moss_tts_delay": (".moss_delay", "generate_moss_tts_delay"),
    "synthesize_moss_tts_delay_conversations": (".moss_delay", "synthesize_moss_tts_delay_conversations"),
    "MossTTSLocalBatchSynthesisOutput": (".moss_local", "MossTTSLocalBatchSynthesisOutput"),
    "MossTTSLocalGenerationConfig": (".moss_local", "MossTTSLocalGenerationConfig"),
    "MossTTSLocalGenerationOutput": (".moss_local", "MossTTSLocalGenerationOutput"),
    "MossTTSLocalSynthesisOutput": (".moss_local", "MossTTSLocalSynthesisOutput"),
    "extract_audio_code_sequences": (".moss_local", "extract_audio_code_sequences"),
    "generate_moss_tts_local": (".moss_local", "generate_moss_tts_local"),
    "sample_next_token": (".moss_local", "sample_next_token"),
    "synthesize_moss_tts_local": (".moss_local", "synthesize_moss_tts_local"),
    "synthesize_moss_tts_local_conversations": (".moss_local", "synthesize_moss_tts_local_conversations"),
    "StepAudioEditXModel": (".step_audio_editx", "StepAudioEditXModel"),
    "StepAudioEditXResult": (".step_audio_editx", "StepAudioEditXResult"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
