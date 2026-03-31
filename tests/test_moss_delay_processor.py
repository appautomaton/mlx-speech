import mlx.core as mx

from mlx_voice.models.moss_delay import MossTTSDelayConfig, MossTTSDelayProcessor


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
