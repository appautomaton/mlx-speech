"""mlx-speech: MLX-native speech library for Apple Silicon."""

__all__ = ["__version__", "tts", "asr", "align"]

__version__ = "0.5.6"


def __getattr__(name: str):
    if name == "tts":
        import mlx_speech.tts as _tts

        return _tts
    if name == "asr":
        import mlx_speech.asr as _asr

        return _asr
    if name == "align":
        import mlx_speech.align as _align

        return _align
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
