import mlx.core as mx
import mlx.nn as nn

from mlx_voice.models.moss_delay import (
    MossTTSDelayConfig,
    MossTTSDelayModel,
    QuantizationConfig,
    load_moss_tts_delay_model,
    quantize_moss_tts_delay_model,
    save_moss_tts_delay_model,
)


def _quantizable_config() -> MossTTSDelayConfig:
    return MossTTSDelayConfig.from_dict(
        {
            "n_vq": 2,
            "audio_vocab_size": 16,
            "language_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 256,
                "max_position_embeddings": 64,
            },
        }
    )


def test_save_and_load_quantized_moss_tts_delay_round_trip(tmp_path) -> None:
    config = _quantizable_config()
    model = MossTTSDelayModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_tts_delay_model(model, quantization)
    save_moss_tts_delay_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )

    loaded = load_moss_tts_delay_model(tmp_path, prefer_mlx_int8=False, strict=True)

    assert loaded.quantization == quantization
    assert loaded.alignment_report.is_exact_match
    assert isinstance(loaded.model.language_model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(loaded.model.lm_heads[0], nn.QuantizedLinear)


def test_quantized_delay_round_trip_model_runs_forward(tmp_path) -> None:
    config = _quantizable_config()
    model = MossTTSDelayModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_tts_delay_model(model, quantization)
    save_moss_tts_delay_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )
    loaded = load_moss_tts_delay_model(tmp_path, prefer_mlx_int8=False, strict=True)

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
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    output = loaded.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    assert output.last_hidden_state.shape == (1, 3, 128)
    assert output.logits_all[0].shape == (1, 3, 256)
    assert output.logits_all[1].shape == (1, 3, 17)
    assert output.last_hidden_state.dtype == mx.bfloat16
    assert output.logits_all[0].dtype == mx.bfloat16
