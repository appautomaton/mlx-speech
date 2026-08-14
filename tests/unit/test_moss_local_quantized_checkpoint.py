import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_speech.models.moss_local import (
    MossTTSLocalConfig,
    load_moss_tts_local_model,
    quantize_moss_tts_local_model,
    save_moss_tts_local_model,
)
from mlx_speech.models.moss_local.checkpoint import QuantizationConfig, prepare_runtime_state_dict
from mlx_speech.models.moss_local.model import MossTTSLocalModel


def _quantizable_config() -> MossTTSLocalConfig:
    return MossTTSLocalConfig.from_dict(
        {
            "n_vq": 2,
            "audio_vocab_size": 16,
            "local_hidden_size": 128,
            "local_ffn_hidden_size": 256,
            "local_num_layers": 2,
            "additional_mlp_ffn_hidden_size": 128,
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


def test_save_and_load_quantized_moss_local_round_trip(tmp_path) -> None:
    config = _quantizable_config()
    model = MossTTSLocalModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_tts_local_model(model, quantization)
    save_moss_tts_local_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )

    loaded = load_moss_tts_local_model(tmp_path, prefer_mlx_int8=False, strict=True)

    assert loaded.quantization == quantization
    assert loaded.alignment_report.is_exact_match
    assert isinstance(loaded.model.model.embedding_list[0], nn.QuantizedEmbedding)
    assert isinstance(loaded.model.lm_heads[0], nn.QuantizedLinear)


def test_quantized_round_trip_model_runs_forward(tmp_path) -> None:
    config = _quantizable_config()
    model = MossTTSLocalModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_tts_local_model(model, quantization)
    save_moss_tts_local_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )
    loaded = load_moss_tts_local_model(tmp_path, prefer_mlx_int8=False, strict=True)

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
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.int32)

    global_output = loaded.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    local_inputs = loaded.model.project_global_to_local(global_output.last_hidden_state)
    local_output = loaded.model.forward_local_sequence(
        local_inputs,
        attention_mask=attention_mask,
    )
    logits = loaded.model.project_local_outputs_to_logits(local_output.last_hidden_state)

    assert global_output.last_hidden_state.shape == (1, 3, 128)
    assert local_output.last_hidden_state.shape == (1, 3, 128)
    assert logits[0].shape == (1, 3, 256)
    assert global_output.last_hidden_state.dtype == mx.bfloat16
    assert local_output.last_hidden_state.dtype == mx.bfloat16
    assert logits[0].dtype == mx.bfloat16


def test_original_runtime_state_dict_preserves_source_dtype() -> None:
    config = _quantizable_config()
    model = MossTTSLocalModel(config)
    state_dict = tree_flatten(model.parameters(), destination={})
    bf16_state = {key: value.astype(mx.bfloat16) for key, value in state_dict.items()}

    runtime_state = prepare_runtime_state_dict(bf16_state, quantization=None)

    assert runtime_state["model.embedding_list.0.weight"].dtype == mx.bfloat16
    assert runtime_state["lm_heads.0.weight"].dtype == mx.bfloat16


def test_quantized_runtime_state_dict_preserves_quantized_dtypes() -> None:
    config = _quantizable_config()
    model = MossTTSLocalModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_tts_local_model(model, quantization)
    state_dict = tree_flatten(model.parameters(), destination={})
    runtime_state = prepare_runtime_state_dict(state_dict, quantization=quantization)

    assert runtime_state["model.embedding_list.0.weight"].dtype != mx.float32
