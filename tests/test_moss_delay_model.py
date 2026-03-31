import mlx.core as mx

from mlx_voice.models.moss_delay import MossTTSDelayConfig, MossTTSDelayModel


def _tiny_config() -> MossTTSDelayConfig:
    return MossTTSDelayConfig.from_dict(
        {
            "n_vq": 2,
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


def test_prepare_multi_modal_inputs_has_expected_shape() -> None:
    model = MossTTSDelayModel(_tiny_config())
    input_ids = mx.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ],
        dtype=mx.int32,
    )

    embeds = model._compute_input_embeddings(input_ids)

    assert embeds.shape == (1, 3, 32)
    assert embeds.dtype == mx.float32


def test_delay_model_forward_returns_expected_logit_shapes() -> None:
    config = _tiny_config()
    model = MossTTSDelayModel(config)
    input_ids = mx.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ]
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array([[1, 1, 1, 1]], dtype=mx.int32)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    assert output.last_hidden_state.shape == (1, 4, 32)
    assert len(output.logits_all) == config.channels
    assert output.logits_all[0].shape == (1, 4, config.vocab_size)
    assert output.logits_all[1].shape == (1, 4, config.audio_embedding_vocab_size)
    assert output.hidden_states is not None


def test_delay_model_prefill_and_decode_step_update_global_cache() -> None:
    config = _tiny_config()
    model = MossTTSDelayModel(config)
    input_ids = mx.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ]
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array([[1, 1, 1, 1]], dtype=mx.bool_)

    prefill_output, kv_cache = model.prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_cache_len=12,
    )

    assert prefill_output.last_hidden_state.shape == (1, 4, 32)
    assert kv_cache.current_length == 4
    assert kv_cache.prompt_length == 4

    next_row = mx.array([[[13, 14, 15]]], dtype=mx.int32)
    decode_output = model.decode_step(
        input_ids=next_row,
        kv_cache=kv_cache,
    )

    assert decode_output.last_hidden_state.shape == (1, 1, 32)
    assert len(decode_output.logits_all) == config.channels
    assert kv_cache.current_length == 5
