"""Tests for Step-Audio-EditX Step1 core."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    Step1Config,
    Step1ForCausalLM,
    Step1KVCacheCollection,
    Step1LayerKVCache,
    _repeat_kv_groups,
    build_sqrt_alibi_bias,
    load_step_audio_editx_checkpoint,
    load_step_audio_editx_model,
    resolve_step_audio_editx_model_dir,
    validate_checkpoint_against_model,
)

MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
HAS_CHECKPOINT = (MODEL_DIR / "config.json").exists()


class TestStep1Config:
    def test_parses_shipped_config(self):
        cfg = Step1Config.from_path(MODEL_DIR)
        assert cfg.hidden_size == 3072
        assert cfg.intermediate_size == 8192
        assert cfg.num_attention_heads == 48
        assert cfg.num_attention_groups == 4
        assert cfg.num_hidden_layers == 32
        assert cfg.vocab_size == 74752
        assert cfg.rms_norm_eps == pytest.approx(1e-5)
        assert cfg.bos_token_id == 1
        assert cfg.pad_token_id == 0
        assert cfg.eos_token_id == 3
        assert cfg.head_dim == 64
        assert cfg.kv_repeat == 12


class TestAlibi:
    def test_bias_shape_and_causality(self):
        bias = build_sqrt_alibi_bias(3, 5, 4, dtype=mx.float32)
        assert bias.shape == (4, 3, 5)
        assert float(bias[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert bias[0, 0, 1].item() == float("-inf")
        assert bias[0, 1, 2].item() == float("-inf")

    def test_bias_matches_closed_form(self):
        bias = build_sqrt_alibi_bias(3, 3, 4, dtype=mx.float32)
        n = 4
        m0 = 2.0 ** (-8.0 / n)
        slopes = [m0**i for i in range(1, n + 1)]
        expected = []
        for slope in slopes:
            rows = []
            for i in range(3):
                row = []
                for j in range(3):
                    if j > i:
                        row.append(float("-inf"))
                    else:
                        row.append(-(i - j) ** 0.5 * slope)
                rows.append(row)
            expected.append(rows)
        expected = mx.array(expected, dtype=mx.float32)
        assert mx.allclose(bias, expected, atol=1e-6, rtol=1e-6)


class TestCache:
    def test_layer_cache_append_get_and_reset(self):
        cache = Step1LayerKVCache.allocate(
            batch_size=1,
            max_length=4,
            num_groups=2,
            head_dim=3,
            dtype=mx.float32,
        )
        k1 = mx.arange(12, dtype=mx.float32).reshape(1, 2, 2, 3)
        v1 = (k1 + 100.0).astype(mx.float32)
        cache.append(k1, v1)

        got_k1, got_v1 = cache.get()
        assert cache.current_length == 2
        assert got_k1.shape == (1, 2, 2, 3)
        assert mx.array_equal(got_k1, k1)
        assert mx.array_equal(got_v1, v1)

        k2 = mx.arange(12, 24, dtype=mx.float32).reshape(1, 2, 2, 3)
        v2 = (k2 + 100.0).astype(mx.float32)
        cache.append(k2, v2)
        got_k2, got_v2 = cache.get()
        expected_k = mx.concatenate([k1, k2], axis=1)
        expected_v = mx.concatenate([v1, v2], axis=1)
        assert cache.current_length == 4
        assert mx.array_equal(got_k2, expected_k)
        assert mx.array_equal(got_v2, expected_v)

        cache.reset()
        reset_k, reset_v = cache.get()
        assert cache.current_length == 0
        assert reset_k.shape == (1, 0, 2, 3)
        assert reset_v.shape == (1, 0, 2, 3)

    def test_layer_cache_overflow_raises(self):
        cache = Step1LayerKVCache.allocate(
            batch_size=1,
            max_length=2,
            num_groups=1,
            head_dim=4,
            dtype=mx.float32,
        )
        with pytest.raises(ValueError, match="KV cache overflow"):
            cache.append(
                mx.zeros((1, 3, 1, 4), dtype=mx.float32),
                mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            )

    def test_cache_collection_grows_across_decode_steps(self):
        cfg = Step1Config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_attention_groups=1,
            num_hidden_layers=1,
            vocab_size=32,
        )
        model = Step1ForCausalLM(cfg)
        cache = model.allocate_kv_cache(batch_size=1, max_length=4, dtype=mx.float32)
        x = mx.array([[1, 2, 3]], dtype=mx.int32)
        out1 = model(input_ids=x, cache=cache)
        mx.eval(out1.logits)
        assert out1.logits.shape == (1, 3, 32)
        assert isinstance(out1.cache, Step1KVCacheCollection)
        assert out1.cache.current_length == 3
        key1, value1 = out1.cache.layers[0].get()
        assert key1.shape == (1, 3, 1, 4)
        assert value1.shape == (1, 3, 1, 4)

        next_ids = mx.array([[4]], dtype=mx.int32)
        out2 = model(input_ids=next_ids, cache=out1.cache)
        mx.eval(out2.logits)
        assert out2.cache.current_length == 4
        key2, value2 = out2.cache.layers[0].get()
        assert key2.shape == (1, 4, 1, 4)
        assert value2.shape == (1, 4, 1, 4)

    def test_full_forward_matches_cached_prefill_then_decode(self):
        cfg = Step1Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_attention_groups=2,
            num_hidden_layers=2,
            vocab_size=48,
        )
        model = Step1ForCausalLM(cfg)
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        full = model(input_ids=input_ids)
        mx.eval(full.logits)

        cache = model.allocate_kv_cache(batch_size=1, max_length=4, dtype=mx.float32)
        prefill = model(input_ids=input_ids[:, :3], cache=cache)
        mx.eval(prefill.logits)
        decode = model(input_ids=input_ids[:, 3:], cache=prefill.cache)
        mx.eval(decode.logits)

        assert mx.allclose(decode.logits[:, -1, :], full.logits[:, 3, :], atol=2e-3, rtol=2e-3)
        assert decode.cache.current_length == 4

    def test_greedy_decode_matches_without_cache(self):
        cfg = Step1Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_attention_groups=2,
            num_hidden_layers=2,
            vocab_size=64,
        )
        model = Step1ForCausalLM(cfg)
        prompt = [1, 2, 3]
        max_new_tokens = 4

        cache = model.allocate_kv_cache(batch_size=1, max_length=len(prompt) + max_new_tokens, dtype=mx.float32)
        outputs = model(input_ids=mx.array([prompt], dtype=mx.int32), cache=cache)
        mx.eval(outputs.logits)
        cached_tokens: list[int] = []
        next_logits = outputs.logits[:, -1, :]
        cached_cache = outputs.cache
        for _ in range(max_new_tokens):
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            cached_tokens.append(next_token)
            outputs = model(input_ids=mx.array([[next_token]], dtype=mx.int32), cache=cached_cache)
            mx.eval(outputs.logits)
            cached_cache = outputs.cache
            next_logits = outputs.logits[:, -1, :]

        full_tokens: list[int] = []
        current = list(prompt)
        for _ in range(max_new_tokens):
            outputs = model(input_ids=mx.array([current], dtype=mx.int32))
            mx.eval(outputs.logits)
            next_token = int(mx.argmax(outputs.logits[:, -1, :], axis=-1).item())
            full_tokens.append(next_token)
            current.append(next_token)

        assert cached_tokens == full_tokens

    def test_repeat_kv_groups_matches_repeat_interleave_order(self):
        x = mx.array([[[[0.0], [1.0], [2.0], [3.0]]]], dtype=mx.float32)
        repeated = _repeat_kv_groups(x, 3)
        assert repeated.shape == (1, 1, 12, 1)
        assert np.asarray(repeated).reshape(-1).tolist() == [
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0,
        ]

    def test_grouped_gqa_matches_explicit_repeat_path(self):
        batch_size = 1
        num_groups = 2
        kv_repeat = 3
        query_len = 2
        key_len = 4
        head_dim = 5
        num_heads = num_groups * kv_repeat

        q = mx.arange(batch_size * num_heads * query_len * head_dim, dtype=mx.float32).reshape(
            batch_size, num_heads, query_len, head_dim,
        ) / 100.0
        k = mx.arange(batch_size * num_groups * key_len * head_dim, dtype=mx.float32).reshape(
            batch_size, num_groups, key_len, head_dim,
        ) / 80.0
        v = mx.arange(batch_size * num_groups * key_len * head_dim, dtype=mx.float32).reshape(
            batch_size, num_groups, key_len, head_dim,
        ) / 60.0

        repeated_k = _repeat_kv_groups(k.transpose(0, 2, 1, 3), kv_repeat).transpose(0, 2, 1, 3)
        repeated_v = _repeat_kv_groups(v.transpose(0, 2, 1, 3), kv_repeat).transpose(0, 2, 1, 3)
        repeated_scores = mx.matmul(q, repeated_k.transpose(0, 1, 3, 2))
        repeated_hidden = mx.matmul(repeated_scores, repeated_v)

        grouped_q = q.reshape(batch_size, num_groups, kv_repeat, query_len, head_dim)
        grouped_k = k[:, :, None, :, :]
        grouped_scores = mx.matmul(
            grouped_q,
            grouped_k.transpose(0, 1, 2, 4, 3),
        ).reshape(batch_size, num_heads, query_len, key_len)
        grouped_v = v[:, :, None, :, :]
        grouped_hidden = mx.matmul(
            grouped_scores.reshape(batch_size, num_groups, kv_repeat, query_len, key_len),
            grouped_v,
        ).reshape(batch_size, num_heads, query_len, head_dim)

        assert mx.allclose(grouped_scores, repeated_scores, atol=1e-6, rtol=1e-6)
        assert mx.allclose(grouped_hidden, repeated_hidden, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Step-Audio-EditX checkpoint not available")
class TestRealCheckpoint:
    def test_checkpoint_alignment(self):
        checkpoint = load_step_audio_editx_checkpoint(MODEL_DIR)
        model = Step1ForCausalLM(checkpoint.config)
        report = validate_checkpoint_against_model(model, checkpoint)
        assert report.is_exact_match
        assert checkpoint.key_count == 291

    def test_load_model(self):
        loaded = load_step_audio_editx_model(MODEL_DIR, strict=True, prefer_mlx_int8=False)
        assert loaded.alignment_report.is_exact_match
        assert loaded.model.config.hidden_size == 3072


def test_default_model_dir_prefers_original_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dir = tmp_path / "stepfun" / "step_audio_editx" / "original"
    mlx_int8_dir = tmp_path / "stepfun" / "step_audio_editx" / "mlx-int8"
    original_dir.mkdir(parents=True)
    mlx_int8_dir.mkdir(parents=True)
    (original_dir / "model.safetensors").write_bytes(b"original")
    (mlx_int8_dir / "model.safetensors").write_bytes(b"mlx")

    import mlx_speech.models.step_audio_editx.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "MODELS_ROOT", tmp_path)

    assert resolve_step_audio_editx_model_dir(None, prefer_mlx_int8=False) == original_dir
    assert resolve_step_audio_editx_model_dir(None, prefer_mlx_int8=True) == mlx_int8_dir
