from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.granite_speech_asr import GraniteSpeechAsrModel
from mlx_speech.models.granite_speech_asr.config import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
    GraniteSpeechTextConfig,
)


def _config(*, max_position_embeddings: int = 32) -> GraniteSpeechConfig:
    return GraniteSpeechConfig(
        encoder=GraniteSpeechEncoderConfig(input_dim=8),
        projector=GraniteSpeechProjectorConfig(hidden_size=8, encoder_hidden_size=8),
        text=GraniteSpeechTextConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=max_position_embeddings,
            eos_token_id=9,
        ),
        audio_token_index=7,
    )


@dataclass
class _Output:
    logits: mx.array
    kv_cache: object


class _FakeLanguageModel:
    def __init__(self):
        self.prefill_inputs = None
        self.max_cache_len = None
        self.decode_inputs: list[int] = []
        self.cache = object()

    def prefill(self, *, inputs_embeds, max_cache_len):
        self.prefill_inputs = inputs_embeds
        self.max_cache_len = max_cache_len
        logits = mx.zeros((1, inputs_embeds.shape[1], 16), dtype=mx.float32)
        logits[:, -1, 5] = 10.0
        return _Output(logits=logits, kv_cache=self.cache)

    def decode_step(self, *, input_ids, kv_cache):
        assert kv_cache is self.cache
        token = int(np.array(input_ids).reshape(-1)[0])
        self.decode_inputs.append(token)
        logits = mx.zeros((1, 1, 16), dtype=mx.float32)
        logits[:, -1, 6 if token == 5 else 9] = 10.0
        return _Output(logits=logits, kv_cache=kv_cache)


class _FakeModel:
    def __init__(self):
        self.language_model = _FakeLanguageModel()
        self.last_masked_ids = None
        self.audio_features_called = False

    def get_audio_features(self, input_features):
        self.audio_features_called = True
        assert input_features.shape == (1, 4, 8)
        return mx.ones((1, 2, 8), dtype=mx.float32) * 99.0

    def embed_input_ids(self, input_ids):
        self.last_masked_ids = input_ids
        ids = mx.broadcast_to(input_ids[..., None], (*input_ids.shape, 8))
        return ids.astype(mx.float32)


class _FakeFeatureExtractor:
    sample_rate = 16000

    def __init__(self):
        self.preflight_called = False
        self.extract_called = False

    def preflight_shape(self, sample_count):
        self.preflight_called = True
        assert sample_count == 160

        class Shape:
            audio_tokens = 2

        return Shape()

    def __call__(self, waveform):
        self.extract_called = True
        assert waveform.dtype == np.float32
        return np.zeros((1, 4, 8), dtype=np.float32), 2


class _FakeTokenizer:
    def build_prompt_ids(self, num_audio_tokens, user_prompt=None):
        assert num_audio_tokens == 2
        return [1, 7, 7, 2]

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(i) for i in ids)


def _runtime(*, max_position_embeddings: int = 32):
    model = _FakeModel()
    feature_extractor = _FakeFeatureExtractor()
    runtime = GraniteSpeechAsrModel(
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=_FakeTokenizer(),
        config=_config(max_position_embeddings=max_position_embeddings),
    )
    return runtime, model, feature_extractor


def test_granite_generation_replaces_audio_embeddings_and_greedy_decodes():
    runtime, model, feature_extractor = _runtime()

    result = runtime.transcribe(np.zeros((160,), dtype=np.float32), max_new_tokens=2)

    assert feature_extractor.preflight_called is True
    assert feature_extractor.extract_called is True
    assert model.audio_features_called is True
    np.testing.assert_array_equal(np.array(model.last_masked_ids), np.array([[1, 0, 0, 2]]))
    prefill_inputs = np.array(model.language_model.prefill_inputs)
    assert prefill_inputs.shape == (1, 4, 8)
    assert np.all(prefill_inputs[0, 1:3] == 99.0)
    assert model.language_model.max_cache_len == 6
    assert model.language_model.decode_inputs == [5]
    assert result.text == "5 6"
    assert result.tokens == [5, 6]
    assert result.prompt_tokens == 4


def test_granite_generation_validates_context_before_prefill():
    runtime, model, feature_extractor = _runtime(max_position_embeddings=5)

    with pytest.raises(ValueError, match="exceeds context"):
        runtime.transcribe(np.zeros((160,), dtype=np.float32), max_new_tokens=2)

    assert feature_extractor.preflight_called is True
    assert feature_extractor.extract_called is False
    assert model.audio_features_called is False
    assert model.language_model.prefill_inputs is None
