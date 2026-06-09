"""mlx-speech: MLX-native speech library for Apple Silicon."""

__all__ = ["__version__", "tts", "asr"]

__version__ = "0.4.1"


def __getattr__(name: str):
    if name == "tts":
        import mlx_speech.tts as _tts

        return _tts
    if name == "asr":
        import mlx_speech.asr as _asr

        return _asr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
