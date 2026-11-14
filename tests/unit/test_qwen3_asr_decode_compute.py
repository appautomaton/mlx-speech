"""Per-call decoder compute options: the reference path stays unchanged and the
native / fused options stay within tolerance of it."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.config import Qwen3ASRTextConfig
from mlx_speech.models.qwen3_asr.text_decoder import (
    FAST_DECODE_COMPUTE,
    NATIVE_DECODE_COMPUTE,
    REFERENCE_DECODE_COMPUTE,
    Qwen3ASRDecodeCompute,
    Qwen3ASRTextAttention,
    Qwen3ASRTextForCausalLM,
    Qwen3ASRTextKVCache,
    _fused_attention,
    _make_additive_attention_mask,
    _repeat_kv,
)

NATIVE = Qwen3ASRDecodeCompute(native_linear=True, native_lm_head=True)
FUSED = Qwen3ASRDecodeCompute(fused_attention=True)
ALL = Qwen3ASRDecodeCompute(native_linear=True, native_lm_head=True, fused_attention=True)


def _tiny_config(*, window: int | None = None) -> Qwen3ASRTextConfig:
    return Qwen3ASRTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        extra={
            "tie_word_embeddings": True,
            "use_sliding_window": window is not None,
            "sliding_window": window,
            "max_window_layers": 0,
        },
    )


def _model(
    seed: int = 3, dtype: mx.Dtype = mx.float32, *, window: int | None = None
) -> Qwen3ASRTextForCausalLM:
    mx.random.seed(seed)
    model = Qwen3ASRTextForCausalLM(_tiny_config(window=window))
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def _run(model, embeds, compute, *, last_only=False, steps=(5, 9)):
    config = model.config
    cache = Qwen3ASRTextKVCache.allocate(
        config,
        batch_size=1,
        max_length=int(embeds.shape[1]) + len(steps),
        dtype=embeds.dtype,
    )
    out = model.prefill(
        inputs_embeds=embeds,
        kv_cache=cache,
        compute=compute,
        last_logits_only=last_only,
    )
    logits = [out.logits[0, -1]]
    for token in steps:
        step = model.decode_step(
            input_ids=mx.array([[token]], dtype=mx.int32),
            kv_cache=cache,
            compute=compute,
        )
        logits.append(step.logits[0, -1])
    mx.eval(logits)
    return out, [np.asarray(value) for value in logits]


def _embeds(seed: int = 11, length: int = 7, dtype: mx.Dtype = mx.float32) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, length, 16)).astype(dtype)


def test_none_and_reference_compute_are_bitwise_identical() -> None:
    model = _model()
    embeds = _embeds()
    _, default = _run(model, embeds, None)
    _, reference = _run(model, embeds, REFERENCE_DECODE_COMPUTE)
    for left, right in zip(default, reference, strict=True):
        np.testing.assert_array_equal(left, right)


def test_native_linear_is_bitwise_reference_for_float32_weights() -> None:
    # With float32 weights the native matmul is the reference matmul.
    model = _model()
    embeds = _embeds()
    _, reference = _run(model, embeds, None)
    _, native = _run(model, embeds, NATIVE)
    for left, right in zip(reference, native, strict=True):
        np.testing.assert_array_equal(left, right)


def test_native_linear_with_bf16_weights_stays_close_to_reference() -> None:
    model = _model(dtype=mx.bfloat16)
    embeds = _embeds()
    _, reference = _run(model, embeds, None)
    _, native = _run(model, embeds, ALL)
    for left, right in zip(reference, native, strict=True):
        scale = max(1.0, float(np.max(np.abs(left))))
        assert float(np.max(np.abs(left - right))) <= 0.05 * scale


def test_fused_decoder_matches_reference_for_float32_weights() -> None:
    model = _model()
    embeds = _embeds()
    _, reference = _run(model, embeds, None)
    _, fused = _run(model, embeds, FUSED)
    for left, right in zip(reference, fused, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-5)


def test_full_forward_default_compute_is_the_reference_path() -> None:
    model = _model()
    embeds = _embeds()
    default = model.model(inputs_embeds=embeds).last_hidden_state
    reference = model.model(inputs_embeds=embeds, compute=REFERENCE_DECODE_COMPUTE).last_hidden_state
    np.testing.assert_array_equal(np.array(default), np.array(reference))


def test_full_forward_fast_compute_matches_reference() -> None:
    # The forced aligner's single cache-free pass runs with FAST_DECODE_COMPUTE.
    model = _model()
    embeds = _embeds(length=23)
    reference = model.model(inputs_embeds=embeds).last_hidden_state
    fast = model.model(inputs_embeds=embeds, compute=FAST_DECODE_COMPUTE).last_hidden_state
    np.testing.assert_allclose(np.array(reference), np.array(fast), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("compute", [FAST_DECODE_COMPUTE, NATIVE_DECODE_COMPUTE])
def test_full_forward_matches_prefill_under_the_same_compute(compute) -> None:
    model = _model()
    embeds = _embeds(length=12)
    cache = Qwen3ASRTextKVCache.allocate(model.config, batch_size=1, max_length=12)
    prefill = model.model.prefill(
        inputs_embeds=embeds, kv_cache=cache, compute=compute
    ).last_hidden_state
    full = model.model(inputs_embeds=embeds, compute=compute).last_hidden_state
    np.testing.assert_allclose(np.array(prefill), np.array(full), rtol=1e-5, atol=1e-6)


def test_last_logits_only_is_the_last_row_of_full_logits() -> None:
    model = _model()
    embeds = _embeds()
    full, _ = _run(model, embeds, None, steps=())
    last, _ = _run(model, embeds, None, last_only=True, steps=())
    assert last.logits.shape == (1, 1, 32)
    # Metal matmul accumulates differently for 1-row vs N-row shapes, so this is
    # a tolerance check rather than bitwise equality.
    np.testing.assert_allclose(
        np.asarray(last.logits[:, -1]), np.asarray(full.logits[:, -1]), rtol=1e-2, atol=1e-2
    )


def _reference_attention(query, key, value, *, scale, query_offset, attention_mask=None):
    key = _repeat_kv(key, query.shape[1] // key.shape[1])
    value = _repeat_kv(value, query.shape[1] // value.shape[1])
    scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * scale
    mask = _make_additive_attention_mask(
        query_len=int(query.shape[2]),
        key_len=int(key.shape[2]),
        query_offset=query_offset,
        dtype=mx.float32,
        attention_mask=attention_mask,
        use_causal_mask=True,
    )
    return mx.matmul(mx.softmax(scores + mask, axis=-1), value)


@pytest.mark.parametrize(
    ("query_len", "key_len", "query_offset"),
    [(6, 6, 0), (1, 9, 8), (4, 9, 5), (3, 9, 2)],
)
def test_fused_attention_matches_masked_reference(query_len, key_len, query_offset) -> None:
    mx.random.seed(query_len * 31 + key_len)
    query = mx.random.normal((1, 4, query_len, 4))
    key = mx.random.normal((1, 2, key_len, 4))
    value = mx.random.normal((1, 2, key_len, 4))
    expected = _reference_attention(query, key, value, scale=0.5, query_offset=query_offset)
    actual = _fused_attention(
        query,
        key,
        value,
        scale=0.5,
        attention_mask=None,
        query_offset=query_offset,
        use_causal_mask=True,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_fused_attention_honors_padding_mask() -> None:
    mx.random.seed(5)
    query = mx.random.normal((1, 4, 5, 4))
    key = mx.random.normal((1, 2, 5, 4))
    value = mx.random.normal((1, 2, 5, 4))
    padding = mx.array([[0, 1, 1, 1, 1]], dtype=mx.int32)
    expected = _reference_attention(
        query, key, value, scale=0.5, query_offset=0, attention_mask=padding
    )
    actual = _fused_attention(
        query,
        key,
        value,
        scale=0.5,
        attention_mask=padding,
        query_offset=0,
        use_causal_mask=True,
    )
    # Row 0 sees only a padded key; compare the rows with a valid key.
    np.testing.assert_allclose(
        np.asarray(actual)[:, :, 1:], np.asarray(expected)[:, :, 1:], rtol=1e-5, atol=1e-5
    )


def test_fused_attention_rejects_queries_past_the_keys() -> None:
    query = mx.zeros((1, 4, 3, 4))
    key = mx.zeros((1, 2, 4, 4))
    with pytest.raises(ValueError):
        _fused_attention(
            query,
            key,
            key,
            scale=0.5,
            attention_mask=None,
            query_offset=2,
            use_causal_mask=True,
        )


def _run_batch(model, compute, *, padded):
    cache = Qwen3ASRTextKVCache.allocate(
        model.config, batch_size=2, max_length=8, dtype=model.model.embed_tokens.weight.dtype
    )
    mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]) if padded else None
    positions = mx.array([[99, 99, 99, 0, 1], [0, 1, 2, 3, 4]]) if padded else None
    outputs = [model.prefill(
        input_ids=mx.array([[0, 0, 0, 1, 2], [3, 4, 5, 6, 7]]),
        attention_mask=mask, position_ids=positions, kv_cache=cache, compute=compute,
    )]
    # Exercise both suffix prefill and the single-token decode kernel.
    for tokens, starts in (([[8, 9], [10, 11]], [2, 5]), ([[12], [13]], [4, 7])):
        width = len(tokens[0])
        if padded:
            mask = mx.concatenate([mask, mx.ones((2, width), dtype=mx.int32)], axis=1)
            positions = mx.array(starts)[:, None] + mx.arange(width)[None, :]
        outputs.append(model.decode_step(
            input_ids=mx.array(tokens), attention_mask=mask, position_ids=positions,
            kv_cache=cache, compute=compute,
        ))
    mx.eval([output.logits for output in outputs], [output.last_hidden_state for output in outputs])
    return outputs, cache


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_native_attention_matches_float32_reference_through_decode(padded, window) -> None:
    # CPU float32 avoids Metal's batch-shape-dependent TF32 GEMM rounding.
    with mx.stream(mx.cpu):
        model = _model(window=window)
        expected, _ = _run_batch(model, REFERENCE_DECODE_COMPUTE, padded=padded)
        actual, _ = _run_batch(model, NATIVE_DECODE_COMPUTE, padded=padded)
        for step, (reference, native) in enumerate(zip(expected, actual, strict=True)):
            for name in ("logits", "last_hidden_state"):
                left, right = np.asarray(getattr(reference, name)), np.asarray(getattr(native, name))
                assert np.isfinite(right).all()
                if padded and step == 0:
                    # Padded-query outputs are intentionally discarded.
                    left = np.concatenate([left[0, -2:], left[1]])
                    right = np.concatenate([right[0, -2:], right[1]])
                np.testing.assert_allclose(right, left, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("window", [None, 3])
def test_native_bf16_attention_stays_native_and_close_to_float32_oracle(
    monkeypatch, padded, window
) -> None:
    # Compare identical BF16-rounded weights to a float32 CPU oracle.
    with mx.stream(mx.cpu):
        oracle = _model(dtype=mx.bfloat16, window=window)
        oracle.set_dtype(mx.float32)
        expected, _ = _run_batch(oracle, REFERENCE_DECODE_COMPUTE, padded=padded)
    model = _model(dtype=mx.bfloat16, window=window)
    calls = {"rms_norm": 0, "rope": 0, "scaled_dot_product_attention": 0}

    def trace(name):
        original = getattr(mx.fast, name)

        def checked(*args, **kwargs):
            count = 3 if name == "scaled_dot_product_attention" else 1
            assert all(value.dtype == mx.bfloat16 for value in args[:count])
            result = original(*args, **kwargs)
            assert result.dtype == mx.bfloat16
            calls[name] += 1
            return result

        monkeypatch.setattr(mx.fast, name, checked)

    for name in calls:
        trace(name)
    actual, cache = _run_batch(model, NATIVE_DECODE_COMPUTE, padded=padded)
    assert calls["rms_norm"] == 27  # Input, Q, K, post-attention, and final norms.
    assert calls["rope"] == 12
    assert calls["scaled_dot_product_attention"] >= 6
    for layer in cache.layers:
        assert layer.keys.dtype == layer.values.dtype == mx.bfloat16
        assert bool(mx.all(mx.isfinite(layer.keys)).item())
        assert bool(mx.all(mx.isfinite(layer.values)).item())
    for step, (reference, native) in enumerate(zip(expected, actual, strict=True)):
        assert native.logits.dtype == mx.float32
        assert native.last_hidden_state.dtype == mx.bfloat16
        for name in ("logits", "last_hidden_state"):
            left = np.asarray(getattr(reference, name))
            right = np.asarray(getattr(native, name).astype(mx.float32))
            assert np.isfinite(right).all()
            if padded and step == 0:
                left = np.concatenate([left[0, -2:], left[1]])
                right = np.concatenate([right[0, -2:], right[1]])
            scale = max(1.0, float(np.max(np.abs(left))))
            assert float(np.max(np.abs(right - left))) <= 0.03 * scale


@pytest.mark.parametrize("compute", [REFERENCE_DECODE_COMPUTE, FAST_DECODE_COMPUTE])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_existing_presets_keep_float32_qkv_and_table_rope(monkeypatch, compute, dtype) -> None:
    model = _model(dtype=dtype)

    def unexpected(*args, **kwargs):
        pytest.fail("Existing presets must not use fused norms or RoPE")

    monkeypatch.setattr(mx.fast, "rms_norm", unexpected)
    monkeypatch.setattr(mx.fast, "rope", unexpected)
    attention = model.model.layers[0].self_attn
    qkv = attention._project_qkv(
        _embeds(dtype=dtype), offset=0, native=compute.native_linear,
        position_ids=mx.array([[8, 2, 5, 1, 9, 0, 4]]),
    )
    assert all(value.dtype == mx.float32 for value in qkv)
    _run_batch(model, compute, padded=True)


def test_attention_module_accepts_compute_on_decode() -> None:
    config = _tiny_config()
    attention = Qwen3ASRTextAttention(config, layer_idx=0)
    cache = Qwen3ASRTextKVCache.allocate(config, batch_size=1, max_length=4)
    hidden = _embeds(length=3)
    out = attention.prefill(hidden, layer_cache=cache.layers[0], compute=ALL)
    step = attention.decode_step(hidden[:, :1], layer_cache=cache.layers[0], compute=ALL)
    mx.eval(out, step)
    assert out.shape == (1, 3, 16)
    assert step.shape == (1, 1, 16)
