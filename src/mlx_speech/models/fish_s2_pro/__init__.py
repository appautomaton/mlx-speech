import importlib

from .config import FishAudioDecoderConfig, FishS2ProConfig, FishTextConfig

__all__ = [
    "Conversation",
    "FishS2Codec",
    "FishS2ProConfig",
    "FishTextConfig",
    "FishAudioDecoderConfig",
    "DualARTransformer",
    "FishS2Tokenizer",
    "Message",
    "TextPart",
]


def __getattr__(name: str):
    if name == "DualARTransformer":
        return importlib.import_module(".model", __name__).DualARTransformer
    if name == "FishS2Codec":
        return importlib.import_module(".codec", __name__).FishS2Codec
    if name == "FishS2Tokenizer":
        return importlib.import_module(".tokenizer", __name__).FishS2Tokenizer
    if name in {"Conversation", "Message", "TextPart"}:
        prompt = importlib.import_module(".prompt", __name__)
        return getattr(prompt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
