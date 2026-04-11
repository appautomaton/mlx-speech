from types import SimpleNamespace
import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_speech.generation.fish_s2_pro as fish_generation
from mlx_speech.generation.fish_s2_pro import (
    FishS2ProRuntime,
    generate_fish_s2_pro,
)
from mlx_speech.models.fish_s2_pro import Conversation, Message, TextPart
from mlx_speech.models.fish_s2_pro.codec import MissingCodecAssetError
from mlx_speech.models.fish_s2_pro.config import (
    FishAudioDecoderConfig,
    FishS2ProConfig,
)


class _FakeTokenizer:
    def __init__(self):
        self.semantic_token_ids = {7: 1001, 3: 1009}
        self._ids = {
            "<|im_start|>": 11,
            "<|im_end|>": 12,
            "<|voice|>": 13,
        }

    @property
    def im_start_id(self):
        return self._ids["<|im_start|>"]

    @property
    def im_end_id(self):
        return self._ids["<|im_end|>"]

    def modality_id(self, modality: str) -> int:
        return self._ids[f"<|{modality}|>"]

    def semantic_id(self, code: int) -> int:
        return self.semantic_token_ids[int(code)]

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        del add_special_tokens, kwargs
        return [200 + ord(ch) for ch in text]


class _FakeModel:
    def __init__(self, semantic_token_ids, codebook_tokens):
        self.semantic_token_ids = list(semantic_token_ids)
        self.codebook_tokens = list(codebook_tokens)
        self.calls = []
        self.fast_calls = []

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        token_id = self.semantic_token_ids.pop(0) if self.semantic_token_ids else 0
        vocab_size = max(token_id + 1, 2000)
        logits = mx.full((1, 1, vocab_size), -1e9, dtype=mx.float32)
        logits[:, :, token_id] = 0
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        self.fast_calls.append(previous_codebooks)
        token_id = self.codebook_tokens.pop(0)
        vocab_size = max(token_id + 1, 32)
        logits = mx.full((1, vocab_size), -1e9, dtype=mx.float32)
        logits[:, token_id] = 0
        return logits


class _FakeCodec:
    def __init__(self):
        self.sample_rate = 44100
        self.decoded_codes = None

    def decode(self, codes):
        self.decoded_codes = codes
        return mx.ones((1, 8), dtype=mx.float32)


class _StopOnlyModel:
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id
        self.calls = []
        self.fast_calls = []

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        vocab_size = max(self.stop_token_id + 1, 2000)
        logits = mx.full((1, 1, vocab_size), -1e9, dtype=mx.float32)
        logits[:, :, self.stop_token_id] = 0.0
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        del hidden_state, previous_codebooks
        self.fast_calls.append(True)
        raise AssertionError("fast_forward should not run after stop token")


def _config(num_codebooks: int = 3, eos_token_id: int = 9999) -> FishS2ProConfig:
    return FishS2ProConfig(
        eos_token_id=eos_token_id,
        audio_decoder_config=FishAudioDecoderConfig(num_codebooks=num_codebooks),
    )


class _MaskedSemanticModel:
    def __init__(self):
        self.calls = []
        self.fast_calls = []

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        logits = mx.full((1, 1, 2000), -1e9, dtype=mx.float32)
        logits[:, :, 1234] = 0.0
        logits[:, :, 1009] = -1.0
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        del hidden_state
        self.fast_calls.append(previous_codebooks)
        token_id = 4 if previous_codebooks.shape[1] == 1 else 2
        logits = mx.full((1, 32), -1e9, dtype=mx.float32)
        logits[:, token_id] = 0.0
        return logits


class _SparseSemanticModel:
    def __init__(self):
        self.calls = []
        self.fast_calls = []

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        logits = mx.full((1, 1, 2000), -1e9, dtype=mx.float32)
        logits[:, :, 1008] = 3.0
        logits[:, :, 1009] = -1.0
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        del hidden_state
        self.fast_calls.append(previous_codebooks)
        token_id = 4 if previous_codebooks.shape[1] == 1 else 2
        logits = mx.full((1, 32), -1e9, dtype=mx.float32)
        logits[:, token_id] = 0.0
        return logits


class _RepeatedSemanticModel:
    def __init__(self):
        self.calls = []
        self.fast_calls = []
        self._next_fast_token = 4

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        logits = mx.full((1, 1, 2000), -1e9, dtype=mx.float32)
        logits[:, :, 1001] = 0.0
        logits[:, :, 1009] = -0.1
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        del hidden_state
        self.fast_calls.append(previous_codebooks)
        token_id = self._next_fast_token
        self._next_fast_token = 2 if self._next_fast_token == 4 else 4
        logits = mx.full((1, 32), -1e9, dtype=mx.float32)
        logits[:, token_id] = 0.0
        return logits


class _SemanticAwareFastModel:
    def __init__(self):
        self.calls = []
        self.fast_calls = []

    def __call__(self, cur, **kwargs):
        self.calls.append(cur)
        logits = mx.full((1, 1, 2000), -1e9, dtype=mx.float32)
        logits[:, :, 1009] = 0.0
        hidden_states = mx.ones((1, 1, 4), dtype=mx.float32)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def fast_forward(self, hidden_state, previous_codebooks):
        del hidden_state
        self.fast_calls.append(previous_codebooks)
        logits = mx.full((1, 32), -1e9, dtype=mx.float32)
        tokens = previous_codebooks.tolist()
        if tokens == [[3]]:
            logits[:, 5] = 0.0
        elif tokens == [[3, 5]]:
            logits[:, 1] = 0.0
        else:
            logits[:, 0] = 0.0
        return logits


def test_generate_codes_uses_explicit_semantic_mapping():
    runtime = FishS2ProRuntime(
        model=_FakeModel([1009], [4, 2]),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.tolist() == [[3], [4], [2]]


def test_generate_codes_eos_returns_no_output_frame():
    runtime = FishS2ProRuntime(
        model=_FakeModel([9999], []),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.shape == (3, 0)


def test_generate_codes_stops_on_tokenizer_im_end_id():
    runtime = FishS2ProRuntime(
        model=_StopOnlyModel(12),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(eos_token_id=9999),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.shape == (3, 0)


def test_synthesize_raises_for_zero_frame_generation_before_codec_decode():
    codec = _FakeCodec()
    runtime = FishS2ProRuntime(
        model=_StopOnlyModel(12),
        tokenizer=_FakeTokenizer(),
        codec=codec,
        config=_config(eos_token_id=9999),
    )

    with pytest.raises(ValueError, match="No Fish S2 audio tokens generated"):
        runtime.synthesize("hi", max_new_tokens=1)

    assert codec.decoded_codes is None


def test_semantic_code_from_token_id_rejects_nonsemantic_non_eos_token():
    runtime = FishS2ProRuntime(
        model=_FakeModel([1234], []),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="1234"):
        runtime._semantic_code_from_token_id(1234)


def test_semantic_logit_bias_masks_nonsemantic_logits():
    runtime = FishS2ProRuntime(
        model=_FakeModel([1009], [4, 2]),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=FishS2ProConfig(
            eos_token_id=9999,
            semantic_start_token_id=1001,
            semantic_end_token_id=1009,
            audio_decoder_config=FishAudioDecoderConfig(num_codebooks=3),
        ),
    )
    logits = mx.full((1, 10000), -5.0, dtype=mx.float32)
    logits[:, 12] = 2.0
    logits[:, 1001] = 1.0
    logits[:, 1008] = 4.0
    logits[:, 1009] = 3.0

    biased = runtime._apply_semantic_logit_bias(logits)

    assert mx.isneginf(biased[:, 1008]).item()
    assert mx.isneginf(biased[:, 200]).item()
    assert biased[:, 12].item() == 2.0
    assert mx.isfinite(biased[:, 9999]).item()
    assert biased[:, 1001].item() == 1.0
    assert biased[:, 1009].item() == 3.0


def test_generate_codes_masks_logits_to_semantic_band_and_eos():
    runtime = FishS2ProRuntime(
        model=_MaskedSemanticModel(),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=FishS2ProConfig(
            eos_token_id=9999,
            semantic_start_token_id=1001,
            semantic_end_token_id=1009,
            audio_decoder_config=FishAudioDecoderConfig(num_codebooks=3),
        ),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.tolist() == [[3], [4], [2]]


def test_generate_codes_prefers_tokenizer_semantic_ids_over_larger_nonsemantic_logits():
    runtime = FishS2ProRuntime(
        model=_SparseSemanticModel(),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=FishS2ProConfig(
            eos_token_id=9999,
            semantic_start_token_id=1001,
            semantic_end_token_id=1009,
            audio_decoder_config=FishAudioDecoderConfig(num_codebooks=3),
        ),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.tolist() == [[3], [4], [2]]


def test_generate_codes_avoids_semantic_collapse_with_fallback_sample(monkeypatch):
    runtime = FishS2ProRuntime(
        model=_RepeatedSemanticModel(),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=FishS2ProConfig(
            eos_token_id=9999,
            semantic_start_token_id=1001,
            semantic_end_token_id=1009,
            audio_decoder_config=FishAudioDecoderConfig(num_codebooks=3),
        ),
    )
    sampled_ids = iter([1001, 1009, 4, 2, 1001, 1009, 4, 2])

    monkeypatch.setattr(
        fish_generation,
        "_sample_token_id",
        lambda *args, **kwargs: next(sampled_ids),
        raising=False,
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=2)

    assert codes[0].tolist() == [7, 3]


def test_generate_codes_feeds_semantic_code_into_fast_decoder_context():
    runtime = FishS2ProRuntime(
        model=_SemanticAwareFastModel(),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(),
    )

    codes = runtime._generate_codes(mx.zeros((4, 2), dtype=mx.int32), max_new_tokens=1)

    assert codes.tolist() == [[3], [5], [1]]


def test_synthesize_builds_prompt_and_decodes_codes():
    tokenizer = _FakeTokenizer()
    model = _FakeModel([1001, 9999], [5, 1])
    codec = _FakeCodec()
    runtime = FishS2ProRuntime(
        model=model,
        tokenizer=tokenizer,
        codec=codec,
        config=_config(),
    )

    out = runtime.synthesize("hi", max_new_tokens=2)
    expected_prompt = Conversation(
        [
            Message(role="system", parts=[TextPart("convert the provided text to speech")]),
            Message(role="user", parts=[TextPart("hi")]),
            Message(role="assistant", modality="voice", parts=[], add_im_start=True, add_im_end=False),
        ]
    ).encode_for_inference(tokenizer, runtime.config.audio_decoder_config.num_codebooks)

    assert model.calls[0].tolist() == expected_prompt[None, :, :].tolist()
    assert codec.decoded_codes.tolist() == [[7], [5], [1]]
    assert out.waveform.tolist() == [1.0] * 8
    assert out.sample_rate == 44100
    assert out.generated_tokens == 1


def test_generation_raises_on_missing_codec(monkeypatch):
    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.FishS2ProRuntime.from_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MissingCodecAssetError("missing codec")
        ),
    )

    with pytest.raises(MissingCodecAssetError):
        generate_fish_s2_pro("hello")


def test_generate_requires_explicit_codec_dir_when_default_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_fish_s2_pro("hello", model_dir=tmp_path)


def test_generate_uses_public_sibling_codec_autodiscovery(monkeypatch):
    calls = {}

    class _Runtime:
        def synthesize(self, text, **kwargs):
            calls["text"] = text
            calls["kwargs"] = kwargs
            return "ok"

    def fake_from_dir(model_dir, *, codec_dir=None):
        calls["model_dir"] = model_dir
        calls["codec_dir"] = codec_dir
        return _Runtime()

    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.FishS2ProRuntime.from_dir",
        fake_from_dir,
    )

    out = generate_fish_s2_pro("hello")

    assert out == "ok"
    assert calls["model_dir"] == "models/fish_s2_pro/original"
    assert calls["codec_dir"] is None
    assert calls["text"] == "hello"
    assert calls["kwargs"] == {
        "max_new_tokens": 256,
        "reference_audio": None,
        "reference_text": None,
    }


def test_generate_rejects_nonpositive_max_new_tokens(monkeypatch):
    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.FishS2ProRuntime.from_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not load runtime")
        ),
    )

    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        generate_fish_s2_pro("hello", max_new_tokens=0)


def test_generate_rejects_dead_sampling_knobs():
    with pytest.raises(TypeError):
        generate_fish_s2_pro("hello", temperature=0.2)


def test_generate_script_does_not_expose_dead_sampling_knobs(monkeypatch):
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "generate" / "fish_s2_pro.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_fish_s2_pro_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        "sys.argv",
        ["generate_fish_s2_pro.py", "--text", "hello"],
    )

    args = module.parse_args()

    assert not hasattr(args, "temperature")
    assert not hasattr(args, "top_p")
    assert args.codec_dir is None


def test_runtime_prefers_sibling_codec_dir_when_present(monkeypatch, tmp_path):
    model_dir = tmp_path / "mlx-bf16"
    codec_dir = tmp_path / "codec-mlx"
    model_dir.mkdir()
    codec_dir.mkdir()

    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.load_fish_s2_pro_checkpoint",
        lambda path: SimpleNamespace(config=_config()),
    )
    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.DualARTransformer",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.load_checkpoint_into_model",
        lambda model, checkpoint, strict: None,
    )
    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.FishS2Tokenizer.from_pretrained",
        lambda path: _FakeTokenizer(),
    )

    calls = {}

    def fake_codec_from_dir(path):
        calls["codec_dir"] = path
        return _FakeCodec()

    monkeypatch.setattr(
        "mlx_speech.generation.fish_s2_pro.FishS2Codec.from_dir",
        fake_codec_from_dir,
    )

    FishS2ProRuntime.from_dir(model_dir)

    assert calls == {"codec_dir": codec_dir}


def test_build_generation_prompt_with_clone_includes_vq_codes():
    tokenizer = _FakeTokenizer()
    runtime = FishS2ProRuntime(
        model=_FakeModel([], []),
        tokenizer=tokenizer,
        codec=_FakeCodec(),
        config=_config(),
    )

    # Use code values 7 and 3 which map to semantic_token_ids in _FakeTokenizer
    ref_codes = mx.array([[7, 3, 7, 3], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=mx.int32)
    prompt = runtime._build_generation_prompt(
        "hello",
        reference_text="some reference text",
        reference_codes=ref_codes,
    )

    # Prompt should be (num_codebooks + 1, seq_len)
    assert prompt.ndim == 2
    assert int(prompt.shape[0]) == 4  # 3 codebooks + 1 token row
    # Should be longer than a non-clone prompt
    plain_prompt = runtime._build_generation_prompt("hello")
    assert int(prompt.shape[1]) > int(plain_prompt.shape[1])


def test_synthesize_rejects_partial_clone_args():
    runtime = FishS2ProRuntime(
        model=_FakeModel([], []),
        tokenizer=_FakeTokenizer(),
        codec=_FakeCodec(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="must both be provided"):
        runtime.synthesize("hello", reference_text="text but no audio")

    with pytest.raises(ValueError, match="must both be provided"):
        runtime.synthesize("hello", reference_audio="/fake/path.wav")
