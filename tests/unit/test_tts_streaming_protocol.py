from __future__ import annotations

from mlx_speech.tts import StreamingTTSModel, TTSOutput
from mlx_speech.tts._adapters.dots_tts import DotsTTSAdapter

from tests.unit.test_dots_tts_adapter import _Generator


class _NonStreamingModel:
    def generate(self, text=None, **kwargs):
        del text, kwargs
        raise NotImplementedError


def test_streaming_protocol_is_exported_and_structurally_optional() -> None:
    dots = DotsTTSAdapter(_Generator())

    assert isinstance(dots, StreamingTTSModel)
    assert not isinstance(_NonStreamingModel(), StreamingTTSModel)
    chunks = list(dots.generate_stream("hello"))
    assert chunks
    assert all(isinstance(chunk, TTSOutput) for chunk in chunks)
