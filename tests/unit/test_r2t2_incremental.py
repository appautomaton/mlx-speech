"""Incremental R2T2 streaming on a tiny random Qwen3-ASR graph.

The incremental session must produce the same greedy tokens as the full-refeed
session window by window, including across closed encoder blocks and a mel
floor change.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.asr._adapters.confucius4_r2t2 import (
    FAST_DECODE_COMPUTE,
    Confucius4R2T2Adapter,
    R2T2StreamCompute,
    _greedy_tokens,
    _greedy_tokens_pipelined,
)
from mlx_speech.asr._adapters.confucius4_r2t2_incremental import R2T2IncrementalStep
from mlx_speech.generation.qwen3_asr import Qwen3ASRTranscriber
from mlx_speech.models.confucius4_r2t2.metrics import R2T2StreamTrace
from mlx_speech.models.qwen3_asr import Qwen3ASRFeatureExtractor
from mlx_speech.models.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel
from mlx_speech.models.qwen3_asr.processor import Qwen3ASRProcessor
from mlx_speech.models.qwen3_asr.text_decoder import Qwen3ASRTextKVCache

SR = 16_000
_CJK = 0x4E00


def _config() -> Qwen3ASRConfig:
    audio = Qwen3ASRAudioConfig(
        d_model=16,
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        downsample_hidden_size=4,
        output_dim=16,
        max_source_positions=64,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    text = Qwen3ASRTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        eos_token_id=9,
        extra={"tie_word_embeddings": True},
    )
    return Qwen3ASRConfig(
        thinker_config=Qwen3ASRThinkerConfig(
            audio_config=audio,
            text_config=text,
            audio_token_id=31,
            audio_start_token_id=29,
            audio_end_token_id=30,
        ),
        support_languages=("Chinese", "English"),
    )


class _Tokenizer:
    audio_token = "<|audio_pad|>"
    audio_bos_token = "<|audio_start|>"
    audio_eos_token = "<|audio_end|>"
    audio_token_id = 31
    audio_bos_token_id = 29
    audio_eos_token_id = 30
    eos_token_id = 9
    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": 9,
        audio_token: 31,
        audio_bos_token: 29,
        audio_eos_token: 30,
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        index = 0
        specials = sorted(self._specials, key=len, reverse=True)
        while index < len(text):
            for token in specials:
                if text.startswith(token, index):
                    ids.append(self._specials[token])
                    index += len(token)
                    break
            else:
                code = ord(text[index]) - _CJK
                ids.append(code if 0 <= code < 32 else 20)
                index += 1
        return ids

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        special = set(self._specials.values())
        return "".join(
            chr(_CJK + int(i)) for i in ids if not (skip_special_tokens and int(i) in special)
        )

    def token_to_id(self, token: str) -> int | None:
        return self._specials.get(token)


def _runtime(seed: int = 7) -> Qwen3ASRTranscriber:
    config = _config()
    mx.random.seed(seed)
    model = Qwen3ASRModel(config)
    mx.eval(model.parameters())
    processor = Qwen3ASRProcessor(
        config=config,
        tokenizer=_Tokenizer(),
        feature_extractor=Qwen3ASRFeatureExtractor(sample_rate=SR, n_mels=8),
    )
    return Qwen3ASRTranscriber(
        model=model, processor=processor, config=config, eos_token_ids=(9,)
    )


def _audio(seconds: float, *, loud_at: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    pcm = (0.05 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    if loud_at is not None:
        start = int(loud_at * SR)
        pcm[start : start + SR // 2] *= 8.0
    return pcm


FULL_FP32 = R2T2StreamCompute(decoder=FAST_DECODE_COMPUTE, last_logits_only=True)
INCR_FP32 = R2T2StreamCompute(
    decoder=FAST_DECODE_COMPUTE, last_logits_only=True, incremental=True
)


def _stream(adapter, pcm, compute):
    trace = R2T2StreamTrace()
    session = adapter.stream_session(compute=compute, trace=trace, chunk_ms=400)
    for start in range(0, len(pcm), 4_000):
        session.feed(pcm[start : start + 4_000])
    text = session.finalize().text
    return text, [step.generated_ids for step in trace.steps]


def test_incremental_matches_full_refeed_window_by_window() -> None:
    adapter = Confucius4R2T2Adapter(_runtime())
    pcm = _audio(19.0, loud_at=13.0)
    full_text, full_ids = _stream(adapter, pcm, FULL_FP32)
    incr_text, incr_ids = _stream(adapter, pcm, INCR_FP32)
    assert len(full_ids) == len(incr_ids) > 40
    assert incr_ids == full_ids
    assert incr_text == full_text


def test_incremental_reuses_closed_blocks_and_kv_prefix() -> None:
    runtime = _runtime()
    adapter = Confucius4R2T2Adapter(runtime)
    trace = R2T2StreamTrace()
    session = adapter.stream_session(compute=INCR_FP32, trace=trace, chunk_ms=400)
    pcm = _audio(19.0)
    for start in range(0, len(pcm), 4_000):
        session.feed(pcm[start : start + 4_000])
    late = [s for s in trace.steps if s.sample_end > 17 * SR]
    assert late
    for step in late:
        assert step.counter("encoder_tokens") < step.audio_tokens
        assert step.counter("prefill_positions") < step.prompt_tokens


def test_incremental_mel_matches_offline_extractor() -> None:
    runtime = _runtime()
    step = R2T2IncrementalStep(
        runtime, context="", language=None, compute=INCR_FP32
    )
    extractor = runtime.processor.feature_extractor
    pcm = _audio(3.0, loud_at=2.0)
    for end in (4_000, 16_000, 32_000, 48_000):
        audio = pcm[:end]
        frames = end // 160
        raw = step._update_raw(audio, frames).copy()
        tail = raw[:, step._stable :]
        top = max(step._stable_max, float(tail.max()) if tail.size else -np.inf)
        floor = np.float32(top) - np.float32(8.0)
        ours = ((np.maximum(raw, floor) + 4.0) / 4.0).astype(np.float32)
        expected = extractor._extract_log_mel(audio)
        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-5)


def test_pipelined_greedy_matches_sequential() -> None:
    runtime = _runtime()
    model = runtime.model
    embeds = mx.random.normal((1, 12, 16))

    def prefill():
        cache = Qwen3ASRTextKVCache.allocate(
            model.config.text_config, batch_size=1, max_length=32
        )
        return model.text_decoder.prefill(
            inputs_embeds=embeds, kv_cache=cache, compute=FAST_DECODE_COMPUTE,
            last_logits_only=True,
        )

    for budget in (1, 2, 4, 8):
        sequential = _greedy_tokens(
            model, prefill(), max_new_tokens=budget, eos_token_ids=(),
            decoder_compute=FAST_DECODE_COMPUTE,
        )
        pipelined = _greedy_tokens_pipelined(
            model, prefill(), max_new_tokens=budget, eos_token_ids=(),
            decoder_compute=FAST_DECODE_COMPUTE,
        )
        assert pipelined == sequential
        assert len(pipelined) == budget


def test_kv_cache_truncate_and_reserve() -> None:
    config = _config().text_config
    cache = Qwen3ASRTextKVCache.allocate(config, batch_size=1, max_length=4)
    layer = cache.layers[0]
    keys = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
    for current in cache.layers:
        current.append(keys, keys)
    cache.prompt_length = 3
    cache.reserve(10)
    assert cache.max_length == 10 and cache.current_length == 3
    np.testing.assert_array_equal(np.asarray(layer.get()[0]), np.asarray(keys))
    cache.truncate(1)
    assert cache.current_length == 1 and cache.prompt_length == 1
    with pytest.raises(ValueError):
        cache.truncate(2)
    cache.reserve(5)
    assert cache.max_length == 10
