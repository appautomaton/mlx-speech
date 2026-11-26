"""Drama 3 audio infilling and codec code encoding on a tiny CPU model."""

import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech.generation.fish_drama_3 as generation
from mlx_speech.audio import write_wav
from mlx_speech.generation.fish_drama_3 import RUNAWAY_FACTOR
from mlx_speech.models.fish_drama_3.prompt import (
    MAX_PROMPT_ROWS,
    PromptTooLongError,
    build_prompt,
)
from tests.unit.test_fish_drama_3_batch import PromptTokenizer, prompt_runtime


@pytest.fixture(autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class RecordingCodec:
    sample_rate = 44_100

    def __init__(self, *, encoded_frames: int = 4) -> None:
        self.model = SimpleNamespace(model=SimpleNamespace(frame_length=2048))
        self.encoded_frames = encoded_frames
        self.encode_shapes: list[tuple[int, ...]] = []
        self.decode_calls: list[np.ndarray] = []

    def encode(self, waveform: mx.array) -> mx.array:
        self.encode_shapes.append(tuple(waveform.shape))
        return mx.arange(3 * self.encoded_frames, dtype=mx.int32).reshape(
            3, self.encoded_frames
        )

    def decode(self, codes: mx.array) -> mx.array:
        arr = np.array(codes, dtype=np.int32)
        self.decode_calls.append(arr)
        return mx.ones((1, arr.shape[1] * 4), dtype=mx.float32)


def test_encode_codes_loads_mono_at_codec_rate_and_encode_reference_delegates(tmp_path):
    codec = RecordingCodec(encoded_frames=5)
    rt = prompt_runtime(codec)
    wav_path = tmp_path / "stereo.wav"
    stereo = mx.zeros((2, 22_050), dtype=mx.float32)
    write_wav(wav_path, stereo, sample_rate=22_050)

    codes = rt.encode_codes(wav_path)
    ref = rt.encode_reference(wav_path, "hello", speaker=2)

    assert codes.dtype == np.int32
    assert codes.shape == (3, 5)
    assert len(codec.encode_shapes) == 2
    assert codec.encode_shapes[0] == (44_100,)
    np.testing.assert_array_equal(ref.codes, codes)
    assert ref.text == "<|speaker:2|> hello"


def test_infill_splices_full_codes_decodes_once_and_reuses_system_head():
    codec = RecordingCodec()
    rt = prompt_runtime(codec)
    prefix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    suffix = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.int32)

    first = rt.infill("hello world", prefix, suffix, max_new_tokens=5, seed=0)

    assert first.text == "hello world"
    assert first.codes.dtype == np.int32
    assert first.codes.shape[0] == 3
    assert first.codes.shape[1] > prefix.shape[1] + suffix.shape[1]
    np.testing.assert_array_equal(first.codes[:, : prefix.shape[1]], prefix)
    np.testing.assert_array_equal(first.codes[:, -suffix.shape[1] :], suffix)
    assert len(codec.decode_calls) == 1
    np.testing.assert_array_equal(codec.decode_calls[0], first.codes)
    assert first.waveform.shape == (first.codes.shape[1] * 4,)

    infill_head = rt._head
    assert infill_head is not None
    second = rt.infill("another edit", prefix, suffix, max_new_tokens=5, seed=1)
    assert rt._head is infill_head

    # A cold runtime with the same tokenizer and seed matches bit for bit.
    cold = prompt_runtime(RecordingCodec())
    cold.tokenizer = rt.tokenizer
    cold_second = cold.infill("another edit", prefix, suffix, max_new_tokens=5, seed=1)
    np.testing.assert_array_equal(second.codes, cold_second.codes)

    # Switching back to TTS replaces the cached system-turn head without colliding.
    tts_prompt = build_prompt("plain tts", rt.tokenizer, rt.config)
    rt.generate_frames(
        tts_prompt, max_new_tokens=2, temperature=1.0, top_k=0, top_p=1.0, key=mx.random.key(0)
    )
    assert rt._head is not infill_head


def test_infill_empty_mid_codes_skips_codec_decode(monkeypatch):
    class ForbidDecodeCodec(RecordingCodec):
        def decode(self, codes: mx.array) -> mx.array:
            pytest.fail("codec.decode must not run when mid_codes is empty")

    rt = prompt_runtime(ForbidDecodeCodec())
    monkeypatch.setattr(
        rt,
        "generate_frames",
        lambda *a, **k: (np.zeros((3, 0), dtype=np.int32), "eos"),
    )
    prefix = np.ones((3, 2), dtype=np.int32)
    suffix = np.full((3, 3), 2, dtype=np.int32)

    chunk = rt.infill("hello", prefix, suffix, seed=0)

    assert chunk.text == "hello"
    assert chunk.codes.shape == (3, 0)
    assert chunk.codes.dtype == np.int32
    assert chunk.waveform.shape == (0,)
    assert chunk.finish_reason == "eos"


def test_infill_duration_seconds_applies_hint_and_budget(monkeypatch):
    codec = RecordingCodec()
    rt = prompt_runtime(codec)
    prefix = np.ones((3, 2), dtype=np.int32)
    suffix = np.full((3, 2), 2, dtype=np.int32)
    seen: dict[str, object] = {}
    real_build = generation.build_infill_prompt

    def capture_prompt(text, pref, suff, tok, cfg):
        seen["prompt_text"] = text
        return real_build(text, pref, suff, tok, cfg)

    def fake_generate(prompt, *, max_new_tokens, temperature, top_k, top_p, key):
        seen["max_new_tokens"] = max_new_tokens
        return np.full((3, 4), 5, dtype=np.int32), "eos"

    monkeypatch.setattr(generation, "build_infill_prompt", capture_prompt)
    monkeypatch.setattr(rt, "generate_frames", fake_generate)

    chunk = rt.infill("hello", prefix, suffix, duration_seconds=0.5, seed=7)

    expected_frames = round(0.5 * 44_100 / 2048)
    assert seen["prompt_text"] == f"hello [audio tokens: {expected_frames}]"
    assert seen["max_new_tokens"] == math.ceil(expected_frames * RUNAWAY_FACTOR)
    assert chunk.text == "hello"
    assert chunk.codes.shape == (3, 8)


def test_infill_validates_inputs_before_generating(monkeypatch):
    rt = prompt_runtime(RecordingCodec())
    monkeypatch.setattr(
        rt, "generate_frames", lambda *a, **k: pytest.fail("should not generate")
    )
    prefix = np.ones((3, 2), dtype=np.int32)
    suffix = np.ones((3, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="empty"):
        rt.infill("   ", prefix, suffix, duration_seconds=0.5)
    with pytest.raises(ValueError, match="positive"):
        rt.infill("hi", prefix, suffix, duration_seconds=0.0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        rt.infill("hi", prefix, suffix, max_new_tokens=0)
    with pytest.raises(ValueError, match="top_k"):
        rt.infill("hi", prefix, suffix, top_k=-1)
    with pytest.raises(ValueError, match="seed"):
        rt.infill("hi", prefix, suffix, seed=-1)
    huge = np.ones((3, MAX_PROMPT_ROWS // 2), dtype=np.int32)
    with pytest.raises(PromptTooLongError):
        rt.infill("hi", huge, huge)
    assert isinstance(rt.tokenizer, PromptTokenizer)


def test_codec_window_transformer_extends_freqs_cis_beyond_block_size() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_speech.models.fish_s2_pro.codec_model import ModelArgs, WindowLimitedTransformer

    mx.random.seed(0)
    small_cfg = ModelArgs(block_size=16, n_layer=1, n_head=2, dim=16, intermediate_size=32, head_dim=8)
    large_cfg = ModelArgs(block_size=32, n_layer=1, n_head=2, dim=16, intermediate_size=32, head_dim=8)
    small = WindowLimitedTransformer(small_cfg, input_dim=16, window_size=8, causal=True)
    large = WindowLimitedTransformer(large_cfg, input_dim=16, window_size=8, causal=True)

    # Copy identical weights into both models (skipping precomputed freqs_cis/causal_mask buffers)
    weights = [
        (k, v)
        for k, v in tree_flatten(small.parameters())
        if not k.endswith(("freqs_cis", "causal_mask"))
    ]
    large.load_weights(weights, strict=False)

    x = mx.random.normal((1, 16, 24))
    y_small = small(x)
    y_large = large(x)
    mx.eval(y_small, y_large)
    assert np.allclose(np.array(y_small), np.array(y_large), atol=1e-5)
    # Precomputed buffer shape on `small` stays intact for strict weight validation
    assert small.freqs_cis.shape[0] == 16


def test_sampling_validation_duration_hint_and_generate_truncation_warning() -> None:
    import math
    import warnings

    import mlx.core as mx
    import pytest

    from mlx_speech.generation.fish_drama_3 import (
        Drama3Chunk,
        _check_sampling,
        _check_sequence_limit,
    )
    from mlx_speech.models.fish_drama_3.prompt import with_duration_hint
    from mlx_speech.tts._adapters.fish_drama_3 import FishDrama3Adapter

    assert with_duration_hint("Hello world.\n\n", 42) == "Hello world. [audio tokens: 42]"

    _check_sampling(0.0, 0, 1.0, 1)
    for bad in (
        (-0.1, 0, 1.0, 10),
        (math.nan, 0, 1.0, 10),
        (1.0, 0, 0.0, 10),
        (1.0, 0, 1.5, 10),
        (1.0, 0, math.nan, 10),
        (1.0, -1, 1.0, 10),
        (1.0, 0, 1.0, 0),
    ):
        with pytest.raises(ValueError):
            _check_sampling(*bad)

    with pytest.raises(ValueError, match="max_position_embeddings"):
        _check_sequence_limit(100, 50, 120)

    class _StubRuntime:
        sample_rate = 44100

        def synthesize(self, *args, **kwargs):
            yield Drama3Chunk(
                text="cut",
                codes=np.zeros((10, 4), dtype=np.int32),
                waveform=mx.zeros((100,), dtype=mx.float32),
                finish_reason="length",
            )

    adapter = FishDrama3Adapter(_StubRuntime())  # type: ignore[arg-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = adapter.generate("cut")
    assert out.waveform.shape == (100,)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
