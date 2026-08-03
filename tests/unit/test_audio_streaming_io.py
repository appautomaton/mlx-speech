from __future__ import annotations

import wave

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.audio import write_wav_chunks


def test_write_wav_chunks_writes_one_mono_stream(tmp_path) -> None:
    output = write_wav_chunks(
        tmp_path / "stream.wav",
        (mx.array([0.0, 0.5]), mx.array([-0.5, 1.5, -1.5])),
        sample_rate=48_000,
    )
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 5
        samples = np.frombuffer(wav_file.readframes(5), dtype=np.int16)
    assert samples.tolist() == [0, 16383, -16383, 32767, -32767]


@pytest.mark.parametrize(
    ("chunks", "message"),
    (
        ((), "empty waveform"),
        ((mx.zeros((2, 2)),), "mono waveform"),
        ((mx.array([float("nan")]),), "non-finite"),
    ),
)
def test_write_wav_chunks_rolls_back_invalid_stream(tmp_path, chunks, message) -> None:
    output = tmp_path / "stream.wav"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match=message):
        write_wav_chunks(output, chunks, sample_rate=48_000)
    assert output.read_bytes() == b"existing"
    assert list(tmp_path.glob(".stream.wav.*.tmp")) == []


def test_write_wav_chunks_rolls_back_iterator_failure(tmp_path) -> None:
    output = tmp_path / "stream.wav"

    def chunks():
        yield mx.ones((4,))
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        write_wav_chunks(output, chunks(), sample_rate=48_000)
    assert not output.exists()
    assert list(tmp_path.glob(".stream.wav.*.tmp")) == []
