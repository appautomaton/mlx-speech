import mlx.core as mx
from pathlib import Path
import sys

import pytest

from mlx_voice.models.moss_delay import MossTTSDelayConfig, MossTTSDelayProcessor
from mlx_voice.models.moss_local.tokenizer import DEFAULT_MOSS_CHAT_TEMPLATE


class _DummyTokenizer:
    def token_to_id(self, token: str) -> int:
        if token == "<|im_start|>":
            return 151644
        if token == "<|im_end|>":
            return 151645
        raise KeyError(token)


def _tiny_config() -> MossTTSDelayConfig:
    return MossTTSDelayConfig.from_dict(
        {
            "n_vq": 3,
            "audio_vocab_size": 16,
            "audio_pad_code": 16,
            "language_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 128,
                "max_position_embeddings": 64,
            },
        }
    )


def _make_reference_codes(length: int = 4, n_vq: int = 16) -> mx.array:
    rows = []
    for frame in range(length):
        rows.append([(frame + column) % 1024 for column in range(n_vq)])
    return mx.array(rows, dtype=mx.int32)


def _load_upstream_delay_processor():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = repo_root / ".references" / "MOSS-TTS"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig as UpstreamConfig
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor as UpstreamProcessor

    model_dir = repo_root / "models" / "openmoss" / "moss_ttsd" / "original"
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.chat_template = DEFAULT_MOSS_CHAT_TEMPLATE
    config = UpstreamConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = UpstreamProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )
    return processor, torch


def _assert_processor_parity(
    our_conversation: list[dict],
    upstream_conversation: list[dict],
    *,
    mode: str,
) -> None:
    processor = MossTTSDelayProcessor.from_path("models/openmoss/moss_ttsd/original")
    upstream_processor, _ = _load_upstream_delay_processor()

    our_batch = processor([our_conversation], mode=mode)
    upstream_batch = upstream_processor([upstream_conversation], mode=mode)

    assert our_batch.input_ids.tolist() == upstream_batch["input_ids"].tolist()
    assert our_batch.attention_mask.tolist() == upstream_batch["attention_mask"].tolist()


def test_delay_pattern_round_trip_restores_original_codes() -> None:
    config = _tiny_config()
    codes = mx.array([[1, 11, 21], [2, 12, 22], [3, 13, 23]], dtype=mx.int32)

    delayed = MossTTSDelayProcessor.apply_delay_pattern(codes, pad_code=config.audio_pad_code)
    restored = MossTTSDelayProcessor.apply_de_delay_pattern(delayed)

    assert delayed.shape == (5, 3)
    assert restored.tolist() == codes.tolist()


def test_delay_processor_parse_audio_codes_de_delays_before_decode() -> None:
    config = _tiny_config()
    processor = MossTTSDelayProcessor(tokenizer=_DummyTokenizer(), model_config=config)
    original_codes = mx.array([[4, 14, 24], [5, 15, 25]], dtype=mx.int32)
    delayed_codes = MossTTSDelayProcessor.apply_delay_pattern(
        original_codes,
        pad_code=config.audio_pad_code,
    )
    captured: list[mx.array] = []

    def _fake_decode_audio_codes(audio_tokens_list: mx.array | list[mx.array]) -> list[mx.array]:
        assert isinstance(audio_tokens_list, list)
        captured.extend(audio_tokens_list)
        return [mx.zeros((8,), dtype=mx.float32)]

    processor.decode_audio_codes = _fake_decode_audio_codes  # type: ignore[method-assign]
    output = processor._parse_audio_codes(0, delayed_codes)

    assert len(output) == 1
    assert len(captured) == 1
    assert captured[0].tolist() == original_codes.tolist()


def test_delay_processor_decode_sequences_keeps_audio_when_text_is_empty() -> None:
    processor = MossTTSDelayProcessor(tokenizer=_DummyTokenizer(), model_config=_tiny_config())
    expected_audio = mx.array([0.1, -0.2, 0.3], dtype=mx.float32)

    processor._parse_text_codes = lambda start_length, text_codes: ""  # type: ignore[method-assign]
    processor._parse_audio_codes = lambda start_length, audio_codes: [expected_audio]  # type: ignore[method-assign]

    messages = processor.decode_sequences(
        [(0, mx.zeros((4, processor.model_config.n_vq + 1), dtype=mx.int32))]
    )

    assert len(messages) == 1
    assert messages[0] is not None
    assert messages[0].content == ""
    assert len(messages[0].audio_codes_list) == 1
    assert messages[0].audio_codes_list[0].tolist() == expected_audio.tolist()


def test_delay_processor_generation_parity_with_upstream_direct() -> None:
    processor = MossTTSDelayProcessor.from_path("models/openmoss/moss_ttsd/original")
    our_message = processor.build_user_message(text="[S1] Hello from MLX delay.")
    upstream_processor, _ = _load_upstream_delay_processor()
    upstream_message = upstream_processor.build_user_message(text="[S1] Hello from MLX delay.")

    _assert_processor_parity([our_message], [upstream_message], mode="generation")


def test_delay_processor_generation_parity_with_upstream_clone() -> None:
    processor = MossTTSDelayProcessor.from_path("models/openmoss/moss_ttsd/original")
    reference_codes = _make_reference_codes()
    our_message = processor.build_user_message(text="[S1] Clone this.", reference=[reference_codes])
    upstream_processor, torch = _load_upstream_delay_processor()
    upstream_message = upstream_processor.build_user_message(
        text="[S1] Clone this.",
        reference=[torch.tensor(reference_codes.tolist(), dtype=torch.long)],
    )

    _assert_processor_parity([our_message], [upstream_message], mode="generation")


def test_delay_processor_continuation_parity_with_upstream() -> None:
    processor = MossTTSDelayProcessor.from_path("models/openmoss/moss_ttsd/original")
    prefix_codes = _make_reference_codes()
    our_conversation = [
        processor.build_user_message(text="[S1] Continue this."),
        processor.build_assistant_message(audio_codes_list=[prefix_codes]),
    ]
    upstream_processor, torch = _load_upstream_delay_processor()
    upstream_conversation = [
        upstream_processor.build_user_message(text="[S1] Continue this."),
        upstream_processor.build_assistant_message(
            audio_codes_list=[torch.tensor(prefix_codes.tolist(), dtype=torch.long)]
        ),
    ]

    _assert_processor_parity(our_conversation, upstream_conversation, mode="continuation")
