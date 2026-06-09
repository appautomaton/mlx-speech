from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.qwen3_asr import Qwen3ASRTranscriber
from mlx_speech.models.qwen3_asr import Qwen3ASRFeatureBatch, Qwen3ASRFeatureExtractor
from mlx_speech.models.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from mlx_speech.models.qwen3_asr.processor import Qwen3ASRProcessor


def _config(*, max_position_embeddings: int = 64) -> Qwen3ASRConfig:
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
        eos_token_id=9,
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

    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
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
    def __init__(self):
        self.audio_features_called = False
        self.prefill_inputs = None
        self.decode_inputs: list[int] = []
        self.cache = object()

    def get_audio_features(self, input_features, *, feature_attention_mask):
        self.audio_features_called = True
        assert input_features.shape == (1, 8, 10)
        assert feature_attention_mask.shape == (1, 10)
        return mx.ones((2, 8), dtype=mx.float32) * 99.0

    def embed_input_ids(self, input_ids):
        ids = mx.where(input_ids == 10, mx.array(0, dtype=input_ids.dtype), input_ids)
        return mx.broadcast_to(ids[..., None], (*ids.shape, 8)).astype(mx.float32)

    def prefill(self, *, inputs_embeds, max_cache_len):
        self.prefill_inputs = inputs_embeds
        self.max_cache_len = max_cache_len
        logits = mx.zeros((1, inputs_embeds.shape[1], 32), dtype=mx.float32)
        logits[:, -1, 5] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=self.cache)

    def decode_step(self, *, input_ids, kv_cache):
        assert kv_cache is self.cache
        token = int(np.array(input_ids).reshape(-1)[0])
        self.decode_inputs.append(token)
        logits = mx.zeros((1, 1, 32), dtype=mx.float32)
        logits[:, -1, 6 if token == 5 else 9] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=kv_cache)


def _runtime(*, max_position_embeddings: int = 64):
    config = _config(max_position_embeddings=max_position_embeddings)
    feature_extractor = _FakeFeatureExtractor()
    processor = Qwen3ASRProcessor(
        config=config,
        tokenizer=_FakeTokenizer(),
        feature_extractor=feature_extractor,
    )
    model = _FakeModel()
    return Qwen3ASRTranscriber(model=model, processor=processor, config=config), model, feature_extractor


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


def test_qwen3_asr_generation_validates_context_before_feature_extraction():
    runtime, model, feature_extractor = _runtime(max_position_embeddings=5)

    with pytest.raises(ValueError, match="exceeds context"):
        runtime.transcribe(np.zeros((1600,), dtype=np.float32), max_new_tokens=2)

    assert feature_extractor.preflight_called is True
    assert feature_extractor.extract_called is False
    assert model.audio_features_called is False
    assert model.prefill_inputs is None
