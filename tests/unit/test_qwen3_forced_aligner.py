"""Qwen3-ForcedAligner: text units, timestamp decoding, config, and tiny end-to-end loading.

The ``_upstream_*`` functions are literal ports of ``Qwen3ForceAlignProcessor``
from Qwen3-ASR 7c6daf7 (``qwen_asr/inference/qwen3_forced_aligner.py``) and act
as oracles for the MLX-side rewrites.
"""

from __future__ import annotations

import json
import unicodedata

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_speech
from mlx_speech.align import AlignItem
from mlx_speech.align._registry import _resolve_align_family
from mlx_speech.asr._registry import _resolve_asr_family
from mlx_speech.generation.qwen3_forced_aligner import Qwen3ForcedAligner
from mlx_speech.models.qwen3_asr.checkpoint import (
    QuantizationConfig,
    load_checkpoint_into_model,
    load_qwen3_asr_checkpoint,
    quantize_qwen3_asr_model,
    save_qwen3_asr_model,
)
from mlx_speech.models.qwen3_asr.config import Qwen3ASRConfig, Qwen3ASRTextConfig
from mlx_speech.models.qwen3_asr.forced_aligner import (
    Qwen3ForcedAlignerSettings,
    build_aligner_prompt,
    decode_timestamps,
    fix_timestamps,
    split_text_units,
)
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel
from mlx_speech.models.qwen3_asr.text_decoder import Qwen3ASRTextForCausalLM
import mlx_speech.models.qwen3_asr.tokenizer as tokenizer_module


# --- upstream oracles -------------------------------------------------------


def _upstream_is_kept_char(ch):
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _upstream_is_cjk_char(ch):
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _upstream_tokenize_space_lang(text):
    tokens = []
    for seg in text.split():
        cleaned = "".join(ch for ch in seg if _upstream_is_kept_char(ch))
        if not cleaned:
            continue
        buf = []
        for ch in cleaned:
            if _upstream_is_cjk_char(ch):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
    return tokens


def _upstream_fix_timestamp(data):
    data = data.tolist()
    n = len(data)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    max_length = max(dp)
    max_idx = dp.index(max_length)
    lis_indices = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()
    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True
    result = data.copy()
    i = 0
    while i < n:
        if not is_normal[i]:
            j = i
            while j < n and not is_normal[j]:
                j += 1
            anomaly_count = j - i
            left_val = None
            for k in range(i - 1, -1, -1):
                if is_normal[k]:
                    left_val = result[k]
                    break
            right_val = None
            for k in range(j, n):
                if is_normal[k]:
                    right_val = result[k]
                    break
            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None:
                        result[k] = right_val
                    elif right_val is None:
                        result[k] = left_val
                    else:
                        result[k] = left_val if (k - (i - 1)) <= ((j) - k) else right_val
            else:
                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        result[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        result[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        result[k] = right_val
            i = j
        else:
            i += 1
    return [int(res) for res in result]


# --- text units and prompt --------------------------------------------------


def test_chinese_text_splits_into_characters_without_punctuation():
    assert split_text_units("今天聊三件事：美联储，国债。") == list("今天聊三件事美联储国债")


def test_english_text_keeps_words_and_apostrophes():
    assert split_text_units("Hello, world! It's 3.75% - O'Neill's.") == [
        "Hello",
        "world",
        "It's",
        "375",
        "O'Neill's",
    ]


def test_mixed_text_splits_cjk_from_latin_runs():
    assert split_text_units("而美联储自己盯的核心PCE，已经到了") == [
        *"而美联储自己盯的核心",
        "PCE",
        *"已经到了",
    ]
    assert split_text_units("核心PCE") == ["核", "心", "PCE"]


def test_text_units_match_upstream_rule():
    samples = [
        "",
        "，。！？",
        "大家好。",
        "Hello   world\tagain\nand　again",  # ideographic space
        "AI模型GPT-4o很强，但2026年的Llama也不错",
        "state-of-the-art rock'n'roll, DON'T",
        "５０％的ＡＢＣ",  # full-width digits and letters
        "\U00020000\U00020001 ext-B, \uf900\ufa6e compat, \u3105\u3106 bopomofo",
        "café naïve Ⅻ ½",
    ]
    for text in samples:
        assert split_text_units(text) == _upstream_tokenize_space_lang(text), text


def test_aligner_prompt_layout():
    prompt = build_aligner_prompt(["核", "PCE"], audio_length=3)
    assert prompt == (
        "<|audio_start|>"
        + "<|audio_pad|>" * 3
        + "<|audio_end|>"
        + "核<timestamp><timestamp>PCE<timestamp><timestamp>"
    )


@pytest.mark.parametrize("units, audio_length", [([], 3), (["a"], 0)])
def test_aligner_prompt_rejects_empty_inputs(units, audio_length):
    with pytest.raises(ValueError):
        build_aligner_prompt(units, audio_length=audio_length)


# --- timestamp decoding -----------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], []),
        ([0, 80, 160, 160, 240], [0, 80, 160, 160, 240]),
        ([0, 80, 2000, 240, 320], [0, 80, 80, 240, 320]),  # lone spike snaps left
        ([0, 80, 160, 2000, 320], [0, 80, 160, 2000, 2000]),  # equal lengths: earliest end wins
        ([0, 400, 300, 200, 100, 1600, 2000], [0, 400, 700, 1000, 1300, 1600, 2000]),  # run of 3
        ([4000, 80, 160], [80, 80, 160]),  # leading anomaly takes the right value
    ],
)
def test_fix_timestamps_examples(values, expected):
    assert fix_timestamps(np.asarray(values, dtype=np.float64)) == expected


def test_fix_timestamps_matches_upstream_on_random_sequences():
    rng = np.random.default_rng(20261031)
    for _ in range(400):
        n = int(rng.integers(1, 80))
        classes = np.sort(rng.integers(0, 400, size=n))
        flips = rng.random(n) < rng.uniform(0.0, 0.4)
        classes[flips] = rng.integers(0, 400, size=int(flips.sum()))
        values = classes.astype(np.float32) * np.float32(80.0)  # upstream's float32 product
        assert fix_timestamps(values) == _upstream_fix_timestamp(values)


def test_decode_timestamps_pairs_starts_and_ends_in_seconds():
    spans = decode_timestamps(["大", "家", "好"], [1, 3, 3, 5, 6, 9], segment_time_ms=80)
    assert spans == [("大", 0.08, 0.24), ("家", 0.24, 0.4), ("好", 0.48, 0.72)]


def test_decode_timestamps_rejects_count_mismatch():
    with pytest.raises(ValueError, match="Expected 4 timestamps"):
        decode_timestamps(["a", "b"], [1, 2, 3], segment_time_ms=80)


# --- config, model head, and dispatch ---------------------------------------


AUDIO = dict(
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
EOS_ID, AUDIO_START_ID, AUDIO_END_ID, AUDIO_PAD_ID, TIMESTAMP_ID = 256, 257, 258, 259, 260
CLASSIFY_NUM = 40


def _text(vocab_size=264, max_positions=256, tie=False):
    return dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=vocab_size,
        max_position_embeddings=max_positions,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=tie,
    )


def _aligner_payload(max_positions=256):
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "timestamp_token_id": TIMESTAMP_ID,
        "timestamp_segment_time": 80,
        "thinker_config": {
            "model_type": "qwen3_forced_aligner",
            "classify_num": CLASSIFY_NUM,
            "audio_config": AUDIO,
            "text_config": _text(max_positions=max_positions),
            "audio_token_id": AUDIO_PAD_ID,
            "audio_start_token_id": AUDIO_START_ID,
            "audio_end_token_id": AUDIO_END_ID,
        },
    }


def test_aligner_config_exposes_time_head_and_timestamp_settings():
    config = Qwen3ASRConfig.from_dict(_aligner_payload())

    assert config.thinker_config.is_forced_aligner
    assert config.thinker_config.lm_head_size == CLASSIFY_NUM
    settings = Qwen3ForcedAlignerSettings.from_config(config)
    assert settings == Qwen3ForcedAlignerSettings(TIMESTAMP_ID, 80.0, CLASSIFY_NUM)
    # The converter writes config.json through to_dict.
    assert Qwen3ForcedAlignerSettings.from_config(Qwen3ASRConfig.from_dict(config.to_dict())) == settings


def test_asr_config_keeps_a_vocabulary_head():
    payload = _aligner_payload()
    payload["thinker_config"]["model_type"] = "qwen3_asr"
    config = Qwen3ASRConfig.from_dict(payload)

    assert not config.thinker_config.is_forced_aligner
    assert config.thinker_config.lm_head_size == 264
    with pytest.raises(ValueError, match="Not a Qwen3-ForcedAligner"):
        Qwen3ForcedAlignerSettings.from_config(config)


@pytest.mark.parametrize(
    "drop, message",
    [(("thinker_config", "classify_num"), "classify_num"), ((None, "timestamp_token_id"), "missing")],
)
def test_aligner_config_reports_missing_fields(drop, message):
    payload = _aligner_payload()
    section, key = drop
    (payload[section] if section else payload).pop(key)
    with pytest.raises(ValueError, match=message):
        Qwen3ForcedAlignerSettings.from_config(Qwen3ASRConfig.from_dict(payload))


def test_aligner_model_builds_a_time_class_head():
    model = Qwen3ASRModel(Qwen3ASRConfig.from_dict(_aligner_payload()))
    assert model.text_decoder.lm_head.weight.shape == (CLASSIFY_NUM, 16)


def test_tied_head_must_be_vocabulary_sized():
    text = Qwen3ASRTextConfig.from_dict(_text(tie=True))
    assert Qwen3ASRTextForCausalLM(text).lm_head.weight.shape == (264, 16)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        Qwen3ASRTextForCausalLM(text, lm_head_size=CLASSIFY_NUM)


def test_registries_route_aligner_and_asr_checkpoints(tmp_path):
    aligner_dir = tmp_path / "aligner"
    asr_dir = tmp_path / "asr"
    aligner_dir.mkdir()
    asr_dir.mkdir()
    (aligner_dir / "config.json").write_text(json.dumps(_aligner_payload()), encoding="utf-8")
    asr_payload = _aligner_payload()
    asr_payload["thinker_config"]["model_type"] = "qwen3_asr"
    (asr_dir / "config.json").write_text(json.dumps(asr_payload), encoding="utf-8")

    assert _resolve_align_family(aligner_dir) == "qwen3_forced_aligner"
    assert _resolve_asr_family(asr_dir) == "qwen3"
    with pytest.raises(ValueError, match=r"mlx_speech\.align\.load"):
        _resolve_asr_family(aligner_dir)
    with pytest.raises(ValueError, match="not a forced-alignment checkpoint"):
        _resolve_align_family(asr_dir)


def test_align_load_rejects_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local model path not found"):
        mlx_speech.align.load(tmp_path / "missing")


def test_keep_unquantized_leaves_the_head_in_floating_point(tmp_path):
    payload = _aligner_payload()
    payload["thinker_config"]["audio_config"] = {
        **AUDIO,
        "d_model": 64,
        "output_dim": 64,
        "encoder_ffn_dim": 128,
    }
    payload["thinker_config"]["text_config"].update(
        hidden_size=64, intermediate_size=128, num_attention_heads=8, num_key_value_heads=4, head_dim=8
    )
    config = Qwen3ASRConfig.from_dict(payload)
    quantization = QuantizationConfig(bits=8, group_size=32)
    model = Qwen3ASRModel(config)
    quantize_qwen3_asr_model(model, quantization, keep_unquantized=("text_decoder.lm_head",))

    assert type(model.text_decoder.lm_head) is nn.Linear
    assert isinstance(model.text_decoder.model.layers[0].mlp.up_proj, nn.QuantizedLinear)
    assert isinstance(model.audio_tower.proj1, nn.QuantizedLinear)

    # The load path re-quantizes exactly the saved set.
    save_qwen3_asr_model(model, tmp_path, config=config, quantization=quantization)
    checkpoint = load_qwen3_asr_checkpoint(tmp_path)
    reloaded = Qwen3ASRModel(checkpoint.config)
    quantize_qwen3_asr_model(reloaded, quantization, state_dict=checkpoint.state_dict)
    load_checkpoint_into_model(reloaded, checkpoint, strict=True)
    assert type(reloaded.text_decoder.lm_head) is nn.Linear
    assert isinstance(reloaded.text_decoder.model.layers[0].mlp.up_proj, nn.QuantizedLinear)


# --- tokenizer --------------------------------------------------------------


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _write_tokenizer_assets(model_dir, *, merges=(), extra_vocab=()):
    vocab = {symbol: index for index, symbol in enumerate(sorted(_bytes_to_unicode().values()))}
    for symbol in extra_vocab:
        vocab[symbol] = len(vocab)
    base = len(vocab)
    added = ["<|endoftext|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<timestamp>"]
    tokenizer_config = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "added_tokens_decoder": {
            str(base + i): {"content": token, "special": token != "<timestamp>"}
            for i, token in enumerate(added)
        },
    }
    (model_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (model_dir / "merges.txt").write_text(
        "#version: 0.2\n" + "".join(f"{m}\n" for m in merges), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    return tokenizer_config


def test_qwen2_pretokenizer_keeps_an_apostrophe_with_the_following_letters(tmp_path):
    config = _write_tokenizer_assets(tmp_path, merges=["' N"], extra_vocab=["'N"])
    load = tokenizer_module._load_bpe_tokenizer
    paths = dict(vocab_path=tmp_path / "vocab.json", merges_path=tmp_path / "merges.txt")
    byte_level = load(**paths, tokenizer_config=config)
    qwen2 = load(**paths, tokenizer_config=config, qwen2_pretokenizer=True)

    merged_id = qwen2.token_to_id("'N")
    assert merged_id not in byte_level.encode("O'Neill").ids
    assert merged_id in qwen2.encode("O'Neill").ids
    # NFC: a decomposed "é" encodes like the composed one.
    decomposed, composed = "cafe\u0301", "caf\u00e9"
    assert qwen2.encode(decomposed).ids == qwen2.encode(composed).ids
    assert byte_level.encode(decomposed).ids != byte_level.encode(composed).ids
    assert qwen2.encode("12<timestamp>").ids[-1] == qwen2.token_to_id("<timestamp>")


# --- tiny end-to-end load ---------------------------------------------------


@pytest.fixture(scope="module")
def tiny_aligner_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("tiny_aligner")
    _write_tokenizer_assets(model_dir)
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"feature_size": 8, "sampling_rate": 16000}), encoding="utf-8"
    )
    config = Qwen3ASRConfig.from_dict(_aligner_payload())
    mx.random.seed(0)
    save_qwen3_asr_model(Qwen3ASRModel(config), model_dir, config=config)
    return model_dir


@pytest.fixture(scope="module")
def tiny_aligner(tiny_aligner_dir):
    return mlx_speech.align.load(tiny_aligner_dir)


def test_tiny_aligner_returns_monotonic_items_inside_the_class_range(tiny_aligner):
    audio = np.random.default_rng(0).uniform(-0.1, 0.1, 16000).astype(np.float32)
    items = tiny_aligner.generate(audio, "核心PCE, it's fine.", language="zh")

    assert [item.text for item in items] == ["核", "心", "PCE", "it's", "fine"]
    assert all(isinstance(item, AlignItem) for item in items)
    times = [t for item in items for t in (item.start_s, item.end_s)]
    assert times == sorted(times)
    assert 0.0 <= times[0] and times[-1] <= (CLASSIFY_NUM - 1) * 0.08
    text, start_s, end_s = items[0]
    assert (text, start_s, end_s) == tuple(items[0])


def test_tiny_aligner_head_matches_a_float64_oracle(tiny_aligner_dir):
    runtime = Qwen3ForcedAligner.from_dir(tiny_aligner_dir)
    audio = np.random.default_rng(1).uniform(-0.1, 0.1, 8000).astype(np.float32)
    inputs = runtime.prepare(audio, "hello world", language="en")
    hidden = runtime.timestamp_hidden_states(inputs)
    logits = np.array(runtime.model.text_decoder.project_logits(hidden))

    weight = np.array(runtime.model.text_decoder.lm_head.weight.astype(mx.float32), dtype=np.float64)
    oracle = np.array(hidden.astype(mx.float32), dtype=np.float64) @ weight.T
    np.testing.assert_allclose(logits, oracle, rtol=1e-4, atol=1e-4)
    assert len(inputs.timestamp_positions) == 4


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(language="ja"), ValueError, "exactly 'zh' or 'en'"),
        (dict(text="，。！"), ValueError, "nothing to align"),
        (dict(audio=np.zeros(180 * 16000 + 16, np.float32)), ValueError, "at most 180 s"),
        (dict(sample_rate=44100), ValueError, "16000 Hz"),
        (dict(audio=np.zeros((2, 16000), np.float32)), ValueError, "1-D mono"),
        (dict(text="hello " * 40), ValueError, "the model accepts 256"),
        (dict(audio=np.zeros(0, np.float32)), ValueError, "Audio is empty"),
    ],
)
def test_tiny_aligner_rejects_invalid_requests(tiny_aligner, kwargs, error, message):
    request = dict(audio=np.zeros(16000, np.float32), text="大家好", language="zh")
    request.update(kwargs)
    audio = request.pop("audio")
    text = request.pop("text")
    with pytest.raises(error, match=message):
        tiny_aligner.generate(audio, text, **request)


def test_tiny_aligner_resamples_file_input_to_16k(tiny_aligner_dir, tmp_path):
    import soundfile as sf

    path = tmp_path / "tone.wav"
    t = np.arange(44100) / 44100
    sf.write(path, np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.3, 44100)
    inputs = Qwen3ForcedAligner.from_dir(tiny_aligner_dir).prepare(path, "ok", language="en")
    assert inputs.duration_s == pytest.approx(1.0)


def test_tiny_aligner_rescales_out_of_range_audio_like_upstream(tiny_aligner_dir):
    runtime = Qwen3ForcedAligner.from_dir(tiny_aligner_dir)
    audio = np.random.default_rng(2).uniform(-1.0, 1.0, 16000).astype(np.float32)
    loud = runtime.prepare(audio * 4.0, "ok", language="en").features.input_features
    peak = float(np.max(np.abs(audio)))
    unit = runtime.prepare(audio / peak, "ok", language="en").features.input_features
    np.testing.assert_allclose(loud, unit, atol=1e-5)
