import mlx.core as mx

from mlx_speech.models.moss_local import MosiTTSModel, MossTTSLocalConfig, MossTTSLocalModel


def _tiny_config() -> MossTTSLocalConfig:
    return MossTTSLocalConfig.from_dict(
        {
            "n_vq": 2,
            "audio_vocab_size": 16,
            "local_hidden_size": 12,
            "local_num_layers": 2,
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
    model = MosiTTSModel(_tiny_config())
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

    embeds = model._prepare_multi_modal_inputs(input_ids)

    assert embeds.shape == (1, 3, 32)


def test_global_model_forward_returns_hidden_state_shape() -> None:
    model = MosiTTSModel(_tiny_config())
    input_ids = mx.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [12, 13, 14],
            ],
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=mx.int32,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    assert output.last_hidden_state.shape == (2, 4, 32)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 3


def test_local_transformer_and_heads_have_expected_shapes() -> None:
    config = _tiny_config()
    model = MossTTSLocalModel(config)
    global_hidden = mx.random.normal((2, 5, config.hidden_size))

    local_inputs = model.project_global_to_local(global_hidden)
    assert local_inputs.shape == (2, 5, config.local_hidden_size)

    local_output = model.forward_local_sequence(local_inputs, output_hidden_states=True)
    assert local_output.last_hidden_state.shape == (2, 5, config.local_hidden_size)
    assert local_output.hidden_states is not None

    logits = model.project_local_outputs_to_logits(local_output.last_hidden_state)
    assert len(logits) == config.channels
    assert logits[0].shape == (2, 5, config.vocab_size)
    assert logits[1].shape == (2, 5, config.audio_embedding_vocab_size)
