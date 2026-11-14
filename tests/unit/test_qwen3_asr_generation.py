from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.qwen3_asr import (
    Qwen3ASRTranscriber,
    resolve_eos_token_ids,
)
from mlx_speech.models.qwen3_asr import Qwen3ASRFeatureBatch, Qwen3ASRFeatureExtractor
from mlx_speech.models.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from mlx_speech.models.qwen3_asr.processor import Qwen3ASRProcessor


def _config(
    *,
    max_position_embeddings: int = 64,
    eos_token_id: int | list[int] | None = 9,
) -> Qwen3ASRConfig:
    audio = Qwen3ASRAudioConfig(
        d_model=8,
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        downsample_hidden_size=4,
        output_dim=8,
        max_source_positions=64,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    text = Qwen3ASRTextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        eos_token_id=eos_token_id,
    )
    return Qwen3ASRConfig(
        thinker_config=Qwen3ASRThinkerConfig(
            audio_config=audio,
            text_config=text,
            audio_token_id=10,
            audio_start_token_id=11,
            audio_end_token_id=12,
        ),
        support_languages=("Chinese", "English"),
    )


class _FakeTokenizer:
    audio_token = "<|audio_pad|>"
    audio_bos_token = "<|audio_start|>"
    audio_eos_token = "<|audio_end|>"
    audio_token_id = 10
    audio_bos_token_id = 11
    audio_eos_token_id = 12
    eos_token_id = 9

    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": eos_token_id,
        audio_token: audio_token_id,
        audio_bos_token: audio_bos_token_id,
        audio_eos_token: audio_eos_token_id,
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        index = 0
        special_tokens = sorted(self._specials, key=len, reverse=True)
        while index < len(text):
            for token in special_tokens:
                if text.startswith(token, index):
                    ids.append(self._specials[token])
                    index += len(token)
                    break
            else:
                ids.append(20)
                index += 1
        return ids

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is True
        assert ids == [5, 6]
        return "language Chinese<asr_text>你好 test"

    def token_to_id(self, token: str) -> int | None:
        return self._specials.get(token)


class _FakeFeatureExtractor(Qwen3ASRFeatureExtractor):
    def __init__(self):
        super().__init__(sample_rate=16000, n_mels=8)
        self.preflight_called = False
        self.extract_called = False

    def preflight_shape(self, num_samples: int) -> tuple[int, int]:
        self.preflight_called = True
        assert num_samples == 1600
        return 10, 2

    def __call__(self, audio, *, sample_rate: int = 16000):
        self.extract_called = True
        assert sample_rate == 16000
        return Qwen3ASRFeatureBatch(
            input_features=np.zeros((1, 8, 10), dtype=np.float32),
            feature_attention_mask=np.ones((1, 10), dtype=np.int64),
        )


class _FakeModel:
    def __init__(self, *, eos_token_id: int = 9):
        self.audio_features_called = False
        self.prefill_inputs = None
        self.decode_inputs: list[int] = []
        self.cache = None
        self.eos_token_id = eos_token_id

    def get_audio_features_batch(self, input_features):
        self.audio_features_called = True
        assert len(input_features) == 1
        assert input_features[0].shape == (8, 10)
        return [mx.ones((2, 8), dtype=mx.float32) * 99.0]

    def embed_input_ids(self, input_ids):
        ids = mx.where(input_ids == 10, mx.array(0, dtype=input_ids.dtype), input_ids)
        return mx.broadcast_to(ids[..., None], (*ids.shape, 8)).astype(mx.float32)

    def prefill(self, *, inputs_embeds, max_cache_len, attention_mask=None, position_ids=None,
                compute=None, last_logits_only=False):
        from mlx_speech.models.qwen3_asr.text_decoder import NATIVE_DECODE_COMPUTE
        assert compute is NATIVE_DECODE_COMPUTE
        assert attention_mask is None and position_ids is None
        assert last_logits_only is True
        self.prefill_inputs = inputs_embeds
        self.max_cache_len = max_cache_len
        from mlx_speech.models.qwen3_asr.text_decoder import Qwen3ASRTextKVCache
        self.cache = Qwen3ASRTextKVCache.allocate(
            _config().text_config, batch_size=1, max_length=max_cache_len,
        )
        zeros = mx.zeros((1, 1, inputs_embeds.shape[1], 4))
        self.cache.layers[0].append(zeros, zeros)
        logits = mx.zeros((1, inputs_embeds.shape[1], 32), dtype=mx.float32)
        logits[:, -1, 5] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=self.cache)

    def decode_step(self, *, input_ids, kv_cache, attention_mask=None, position_ids=None, compute=None):
        from mlx_speech.models.qwen3_asr.text_decoder import NATIVE_DECODE_COMPUTE
        assert compute is NATIVE_DECODE_COMPUTE
        assert attention_mask is None and position_ids is None
        assert kv_cache is self.cache
        token = int(np.array(input_ids).reshape(-1)[0])
        self.decode_inputs.append(token)
        zeros = mx.zeros((1, 1, 1, 4))
        self.cache.layers[0].append(zeros, zeros)
        logits = mx.zeros((1, 1, 32), dtype=mx.float32)
        logits[:, -1, 6 if token == 5 else self.eos_token_id] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=kv_cache)


def _runtime(
    *,
    max_position_embeddings: int = 64,
    config_eos_token_id: int | list[int] | None = 9,
    runtime_eos_token_ids: tuple[int, ...] = (),
):
    config = _config(
        max_position_embeddings=max_position_embeddings,
        eos_token_id=config_eos_token_id,
    )
    feature_extractor = _FakeFeatureExtractor()
    processor = Qwen3ASRProcessor(
        config=config,
        tokenizer=_FakeTokenizer(),
        feature_extractor=feature_extractor,
    )
    model = _FakeModel()
    return (
        Qwen3ASRTranscriber(
            model=model,
            processor=processor,
            config=config,
            eos_token_ids=runtime_eos_token_ids,
        ),
        model,
        feature_extractor,
    )


def test_qwen3_asr_generation_replaces_audio_embeddings_and_greedy_decodes():
    runtime, model, feature_extractor = _runtime()

    result = runtime.transcribe(np.zeros((1600,), dtype=np.float32), max_new_tokens=2)

    assert feature_extractor.preflight_called is True
    assert feature_extractor.extract_called is True
    assert model.audio_features_called is True
    prefill_inputs = np.array(model.prefill_inputs)
    audio_positions = np.where(np.all(prefill_inputs[0] == 99.0, axis=-1))[0]
    assert len(audio_positions) == 2
    assert audio_positions[1] == audio_positions[0] + 1
    assert model.max_cache_len == result.prompt_tokens + 2
    assert model.decode_inputs == [5]
    assert result.tokens == [5, 6]
    assert result.language == "Chinese"
    assert result.text == "你好 test"


def test_qwen3_asr_generation_stops_on_resolved_eos_before_budget():
    runtime, model, _feature_extractor = _runtime(
        config_eos_token_id=None,
        runtime_eos_token_ids=(9,),
    )

    result = runtime.transcribe(np.zeros((1600,), dtype=np.float32), max_new_tokens=8)

    assert result.tokens == [5, 6]
    assert len(result.tokens) < 8
    assert model.decode_inputs == [5, 6, 9]  # One speculative step at EOS.


def test_qwen3_asr_resolves_eos_from_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151643, 151645]}),
        encoding="utf-8",
    )

    ids = resolve_eos_token_ids(
        _config(eos_token_id=None),
        _FakeTokenizer(),
        model_dir=tmp_path,
    )

    assert ids == (151643, 151645)


def test_qwen3_asr_resolves_eos_from_tokenizer_specials_when_config_missing():
    ids = resolve_eos_token_ids(_config(eos_token_id=None), _FakeTokenizer())

    assert ids == (2, 9)


def test_qwen3_asr_generation_validates_context_before_feature_extraction():
    runtime, model, feature_extractor = _runtime(max_position_embeddings=5)

    with pytest.raises(ValueError, match="exceeds context"):
        runtime.transcribe(np.zeros((1600,), dtype=np.float32), max_new_tokens=2)

    assert feature_extractor.preflight_called is True
    assert feature_extractor.extract_called is False
    assert model.audio_features_called is False
    assert model.prefill_inputs is None


@pytest.fixture
def tiny_transcriber():
    # Shared real multimodal fixture also used by incremental R2T2 parity tests.
    from tests.unit.test_r2t2_incremental import _runtime

    return _runtime()


def _batch_clips():
    rng = np.random.default_rng(41)
    return [rng.normal(0, 0.05, length).astype(np.float32) for length in [8000, 1600, 6400, 3200, 4800]]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_qwen3_asr_batch_matches_serial_in_input_order(tiny_transcriber, batch_size):
    audios = _batch_clips()
    languages = [None, "Chinese", "auto", "English", ""]
    expected = [tiny_transcriber.transcribe(a, language=lang, max_new_tokens=5)
                for a, lang in zip(audios, languages, strict=True)]
    actual = tiny_transcriber.transcribe_batch(
        audios, language=languages, batch_size=batch_size, max_new_tokens=5
    )
    assert actual == expected


def test_qwen3_asr_group_encodes_once_without_device_position_reads(tiny_transcriber, monkeypatch):
    audios = _batch_clips()
    expected = [tiny_transcriber.transcribe(audio, max_new_tokens=2) for audio in audios]
    encode = tiny_transcriber.model.get_audio_features_batch
    seen = []

    def checked(features):
        seen.append([row.shape for row in features])
        return encode(features)

    def unexpected(*args, **kwargs):
        pytest.fail("batch audio path used the serial encoder or read device positions")

    monkeypatch.setattr(tiny_transcriber.model, "get_audio_features_batch", checked)
    monkeypatch.setattr(tiny_transcriber.model, "get_audio_features", unexpected)
    monkeypatch.setattr("mlx_speech.models.qwen3_asr.model._audio_token_positions", unexpected)
    actual = tiny_transcriber.transcribe_batch(audios, batch_size=8, max_new_tokens=2)

    assert seen == [[(8, length) for length in [10, 20, 30, 40, 50]]]
    assert actual == expected


def test_qwen3_asr_group_rejects_encoded_length_mismatch(monkeypatch):
    runtime, model, _ = _runtime()
    monkeypatch.setattr(model, "get_audio_features_batch", lambda features: [mx.zeros((1, 8))])
    with pytest.raises(RuntimeError, match="encoded audio length does not match"):
        runtime.transcribe(np.zeros(1600, dtype=np.float32), max_new_tokens=2)
    assert model.prefill_inputs is None


@pytest.mark.parametrize("samples", [160, 1600, 7999])
def test_qwen3_asr_prepare_does_not_pad_sub_half_second_audio(tiny_transcriber, samples):
    waveform, prompt = tiny_transcriber._prepare_audio(
        np.zeros(samples, dtype=np.float32), sample_rate=16000, language=None,
        context="", max_new_tokens=2,
    )
    assert len(waveform) == samples
    assert prompt.audio_length == tiny_transcriber.processor.feature_extractor.preflight_shape(samples)[1]


def test_qwen3_asr_batch_broadcasts_string_language(tiny_transcriber):
    audios = _batch_clips()[:3]
    expected = [tiny_transcriber.transcribe(a, language="English", max_new_tokens=2) for a in audios]
    assert tiny_transcriber.transcribe_batch(audios, language="English", max_new_tokens=2) == expected


@pytest.mark.parametrize("audios", ["audio.wav", np.zeros(1600), mx.zeros(1600)])
def test_qwen3_asr_batch_rejects_bare_audio(tiny_transcriber, audios):
    with pytest.raises(TypeError, match="list or tuple"):
        tiny_transcriber.transcribe_batch(audios)


def test_qwen3_asr_batch_rejects_bare_path(tiny_transcriber, tmp_path):
    with pytest.raises(TypeError, match="list or tuple"):
        tiny_transcriber.transcribe_batch(tmp_path / "sample.wav")


def test_qwen3_asr_batch_empty_and_language_count(tiny_transcriber):
    assert tiny_transcriber.transcribe_batch([]) == []
    assert tiny_transcriber.transcribe_batch(()) == []
    with pytest.raises(ValueError, match="language sequence length"):
        tiny_transcriber.transcribe_batch(_batch_clips()[:2], language=["English"])


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_qwen3_asr_batch_validates_group_size(tiny_transcriber, batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive int"):
        tiny_transcriber.transcribe_batch([], batch_size=batch_size)


@pytest.mark.parametrize("budget", [0, -1, 2.5, True])
def test_qwen3_asr_batch_validates_budget(tiny_transcriber, budget):
    with pytest.raises(ValueError, match="max_new_tokens must be a positive int"):
        tiny_transcriber.transcribe_batch([], max_new_tokens=budget)


@pytest.mark.parametrize("bad", [
    "language", "language_type", "path", "stereo", "nan", "empty", "short", "complex",
    "context", "rate", "type",
])
def test_qwen3_asr_batch_preflights_every_row_before_model(tiny_transcriber, monkeypatch, tmp_path, bad):
    def unexpected(*args, **kwargs):
        pytest.fail("model invoked before validation completed")

    monkeypatch.setattr(tiny_transcriber.model, "get_audio_features_batch", unexpected)
    monkeypatch.setattr(tiny_transcriber.model, "prefill", unexpected)
    audios = _batch_clips()[:2]
    languages = [None, None]
    kwargs = {}
    if bad == "language":
        languages[1] = "Klingon"
    elif bad == "language_type":
        languages[1] = 42
    elif bad == "short":
        audios[1] = np.zeros(159, dtype=np.float32)
    elif bad == "complex":
        audios[1] = np.ones(1600, dtype=np.complex64)
    elif bad == "path":
        audios[1] = tmp_path / "missing.wav"
    elif bad == "stereo":
        audios[1] = np.zeros((1600, 2))
    elif bad == "nan":
        audios[1][0] = np.nan
    elif bad == "empty":
        audios[1] = np.array([])
    elif bad == "context":
        # The first row still fits; only the second exceeds the window.
        audios[1] = np.zeros(16000 * 320, dtype=np.float32)
    elif bad == "rate":
        # A path ignores sample_rate, so the error must name row 1 (the array).
        import soundfile as sf
        path = tmp_path / "valid.wav"
        sf.write(path, audios[0], 16000)
        audios[0] = path
        kwargs["sample_rate"] = 8000
    elif bad == "type":
        audios[1] = object()
    with pytest.raises((ValueError, TypeError), match="input 1"):
        tiny_transcriber.transcribe_batch(audios, language=languages, **kwargs)


def test_qwen3_asr_batch_default_budget_is_448(tiny_transcriber, monkeypatch):
    tiny_transcriber.eos_token_ids = (99,)  # Never emitted by the 32-token head.
    prefill = tiny_transcriber.model.prefill

    def checked(**kwargs):
        assert kwargs["max_cache_len"] - kwargs["inputs_embeds"].shape[1] == 64
        return prefill(**kwargs)

    monkeypatch.setattr(tiny_transcriber.model, "prefill", checked)
    audio = _batch_clips()[:1]
    default = tiny_transcriber.transcribe_batch(audio, max_new_tokens=None)
    explicit = tiny_transcriber.transcribe_batch(audio, max_new_tokens=448)
    assert default == explicit
    assert len(default[0].tokens) == 448


def test_qwen3_asr_batch_paths_downmix_resample_and_ignore_rate(tiny_transcriber, tmp_path):
    import soundfile as sf
    from mlx_speech.audio import load_audio

    audio = _batch_clips()[0]
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([audio, audio * 0.5], axis=-1), 24000)
    waveform, _ = load_audio(path, sample_rate=16000, mono=True)
    expected = tiny_transcriber.transcribe(waveform, max_new_tokens=3)
    assert tiny_transcriber.transcribe_batch([path], sample_rate=8000, max_new_tokens=3) == [expected]


def test_qwen3_asr_batch_single_group_uses_unmasked_single_path(tiny_transcriber, monkeypatch):
    original = tiny_transcriber.model.prefill

    def checked(**kwargs):
        assert kwargs["attention_mask"] is None
        assert kwargs["position_ids"] is None
        assert kwargs["last_logits_only"] is True
        return original(**kwargs)

    monkeypatch.setattr(tiny_transcriber.model, "prefill", checked)
    tiny_transcriber.transcribe_batch(_batch_clips()[:1], max_new_tokens=2)



def test_qwen3_asr_batch_compacts_cache_masks_and_positions(tiny_transcriber, monkeypatch):
    mx.random.seed(42)
    tiny_transcriber.model.text_decoder.lm_head.weight = mx.random.normal((32, 16)) * 0.1
    tiny_transcriber.eos_token_ids = (30,)
    audios = _batch_clips()[:4]
    languages = [None, "English", "Chinese", None]
    expected = [tiny_transcriber.transcribe(a, language=lang, max_new_tokens=8)
                for a, lang in zip(audios, languages, strict=True)]
    assert [len(row.tokens) for row in expected] == [3, 3, 2, 3]
    decode = tiny_transcriber.model.decode_step
    sizes = []

    def checked(**kwargs):
        size = kwargs["input_ids"].shape[0]
        sizes.append(size)
        assert kwargs["attention_mask"].shape[0] == size
        assert kwargs["position_ids"].shape == (size, 1)
        assert all(layer.keys.shape[0] == layer.batch_size == size
                   for layer in kwargs["kv_cache"].layers)
        return decode(**kwargs)

    monkeypatch.setattr(tiny_transcriber.model, "decode_step", checked)
    assert tiny_transcriber.transcribe_batch(
        audios, language=languages, batch_size=4, max_new_tokens=8
    ) == expected
    assert sizes == [4, 4, 4, 3]  # One prediction is already in flight at EOS.



def test_qwen3_asr_single_preserves_context_on_shared_decode(tiny_transcriber):
    audio = _batch_clips()[0]
    empty = tiny_transcriber.transcribe(audio, max_new_tokens=2)
    contextual = tiny_transcriber.transcribe(audio, context="meeting vocabulary", max_new_tokens=2)
    assert contextual.prompt_tokens == empty.prompt_tokens + len("meeting vocabulary")
    assert tiny_transcriber.transcribe_batch([audio], max_new_tokens=2) == [empty]



def test_qwen3_asr_batch_accepts_tuple_of_numpy_and_mlx_waveforms(tiny_transcriber):
    audio = _batch_clips()[0]
    single = tiny_transcriber.transcribe(audio, max_new_tokens=2)
    assert tiny_transcriber.transcribe_batch(
        (audio, mx.array(audio)), max_new_tokens=2
    ) == [single, single]


def test_qwen3_asr_batch_default_groups_48_clips(tiny_transcriber, monkeypatch):
    prefill = tiny_transcriber.model.prefill
    sizes = []

    def checked(**kwargs):
        sizes.append(kwargs["inputs_embeds"].shape[0])
        return prefill(**kwargs)

    monkeypatch.setattr(tiny_transcriber.model, "prefill", checked)
    assert len(tiny_transcriber.transcribe_batch(
        [_batch_clips()[1]] * 49, max_new_tokens=1
    )) == 49
    assert sizes == [48, 1]



@pytest.mark.parametrize("extra_survivor", [False, True])
def test_qwen3_asr_pipeline_matches_serial_through_growth_and_compaction(monkeypatch, extra_survivor):
    from tests.unit.test_r2t2_incremental import _runtime
    from mlx_speech.models.qwen3_asr.text_decoder import Qwen3ASRTextKVCache

    # Exact float32 oracle, independent of Metal's batch-dependent TF32 GEMM.
    with mx.stream(mx.cpu):
        runtime = _runtime()
        mx.random.seed(43)
        runtime.model.text_decoder.lm_head.weight = mx.random.normal((32, 16)) * 0.1
        runtime.eos_token_ids = (21,)
        rng = np.random.default_rng(41)
        audios = [rng.normal(0, 0.05, n).astype(np.float32)
                  for n in [800, 1600, 3200, 4800, 6400, 8000, 12000, 16000]]
        languages = [None, "Chinese", "English", None, "Chinese", "English", None, "Chinese"]
        if extra_survivor:
            audios.append(audios[1])
            languages.append("Chinese")
        budget = 150
        from mlx_speech.asr._adapters.confucius4_r2t2 import _greedy_tokens
        from mlx_speech.models.qwen3_asr.text_decoder import FAST_DECODE_COMPUTE

        # Reuse the existing non-pipelined greedy oracle on independent,
        # fully allocated caches, then also compare the public per-row path.
        prefill = runtime.model.prefill
        eager_tokens = []

        def prefill_with_eager_oracle(**kwargs):
            full = {**kwargs, "max_cache_len": kwargs["inputs_embeds"].shape[1] + budget}
            eager_tokens.append(_greedy_tokens(
                runtime.model, prefill(**full), max_new_tokens=budget,
                eos_token_ids=runtime.eos_token_ids, decoder_compute=FAST_DECODE_COMPUTE,
            ))
            return prefill(**kwargs)

        monkeypatch.setattr(runtime.model, "prefill", prefill_with_eager_oracle)
        expected = [runtime.transcribe(audio, language=language, max_new_tokens=budget)
                    for audio, language in zip(audios, languages, strict=True)]
        monkeypatch.setattr(runtime.model, "prefill", prefill)
        assert [result.tokens for result in expected] == eager_tokens
        assert [len(expected[i].tokens) for i in (0, 3, 7)] == [43, 50, 85]
        assert max(len(row.tokens) for row in expected) == budget

        selections = []
        growth = []
        reserve = Qwen3ASRTextKVCache.reserve
        select = Qwen3ASRTextKVCache.select_rows

        def checked_reserve(cache, capacity):
            before = cache.max_length
            # Finished rows must be dropped before growing even below the 1/4 threshold.
            assert all(layer.batch_size <= len(audios) - 2 for layer in cache.layers)
            reserve(cache, capacity)
            growth.append((before - cache.prompt_length, capacity - cache.prompt_length))
            assert all(layer.keys.shape[0] == layer.values.shape[0] == layer.batch_size
                       and layer.keys.shape[2] == layer.values.shape[2] == capacity
                       for layer in cache.layers)

        def checked_select(cache, rows):
            selections.append((cache.current_length - cache.prompt_length, int(rows.size)))
            select(cache, rows)

        monkeypatch.setattr(Qwen3ASRTextKVCache, "reserve", checked_reserve)
        monkeypatch.setattr(Qwen3ASRTextKVCache, "select_rows", checked_select)
        actual = runtime.transcribe_batch(
            audios, language=languages, batch_size=len(audios), max_new_tokens=budget,
        )
        assert actual == expected
        assert growth == [(64, 128), (128, 150)]
        # B=8 reaches 1/4 at the second EOS. B=9 waits until growth instead;
        # both defer the third EOS until the next growth boundary.
        assert selections == [
            (64 if extra_survivor else 51, len(audios) - 2),
            (128, len(audios) - 3),
        ]
