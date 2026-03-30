import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_voice.models.moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    load_moss_audio_tokenizer_model,
    quantize_moss_audio_tokenizer_model,
    sanitize_state_dict,
    save_moss_audio_tokenizer_model,
)
from mlx_voice.models.moss_audio_tokenizer.checkpoint import (
    QuantizationConfig,
    prepare_runtime_state_dict,
)
from mlx_voice.models.moss_audio_tokenizer.model import _apply_codec_rope


def _tiny_codec_config() -> MossAudioTokenizerConfig:
    return MossAudioTokenizerConfig.from_dict(
        {
            "sampling_rate": 16000,
            "sample_rate": 16000,
            "downsample_rate": 1,
            "causal_transformer_context_duration": 1.0,
            "encoder_kwargs": [
                {
                    "module_type": "Transformer",
                    "input_dimension": 1,
                    "output_dimension": 64,
                    "d_model": 64,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 128,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                }
            ],
            "decoder_kwargs": [
                {
                    "module_type": "Transformer",
                    "input_dimension": 64,
                    "output_dimension": 1,
                    "d_model": 64,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 128,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                }
            ],
            "quantizer_type": "rlfq",
            "quantizer_kwargs": {
                "input_dim": 64,
                "rvq_dim": 64,
                "output_dim": 64,
                "num_quantizers": 4,
                "codebook_size": 32,
                "codebook_dim": 64,
                "quantizer_type": "rlfq",
            },
        }
    )


def test_codec_decode_returns_expected_shape() -> None:
    config = _tiny_codec_config()
    model = MossAudioTokenizerModel(config)
    audio_codes = mx.array(
        [
            [[1, 2, 3, 4]],
            [[4, 3, 2, 1]],
            [[0, 1, 0, 1]],
            [[2, 2, 2, 2]],
        ],
        dtype=mx.int32,
    )

    output = model.decode(audio_codes)

    assert output.audio.shape == (1, 1, 4)
    assert tuple(int(v) for v in output.audio_lengths.tolist()) == (4,)


def test_codec_batch_encode_returns_expected_shape() -> None:
    config = _tiny_codec_config()
    model = MossAudioTokenizerModel(config)
    wav_list = [
        mx.linspace(0.0, 0.5, 8, dtype=mx.float32),
        mx.linspace(-0.25, 0.25, 6, dtype=mx.float32),
    ]

    output = model.batch_encode(wav_list)

    assert output.audio_codes.shape[0] == 4
    assert output.audio_codes.shape[1] == 2
    assert tuple(int(v) for v in output.audio_codes_lengths.tolist()) == (8, 6)


def test_codec_encode_decode_round_trip_lengths() -> None:
    config = _tiny_codec_config()
    model = MossAudioTokenizerModel(config)
    wav = mx.sin(mx.arange(0, 16, dtype=mx.float32) * 0.2)

    encoded = model.encode(wav)
    decoded = model.decode(encoded.audio_codes, num_quantizers=int(encoded.audio_codes.shape[0]))

    assert tuple(int(v) for v in encoded.audio_codes_lengths.tolist()) == (16,)
    assert tuple(int(v) for v in decoded.audio_lengths.tolist()) == (16,)
    assert decoded.audio.shape == (1, 1, 16)


def test_codec_rope_matches_pairwise_complex_rotation() -> None:
    q = mx.array(
        [[[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, -2.0]]]],
        dtype=mx.float32,
    )
    k = mx.array(
        [[[[2.0, -1.0, 0.0, 1.0], [1.25, 0.75, -0.5, 0.25]]]],
        dtype=mx.float32,
    )

    q_rot, k_rot = _apply_codec_rope(q, k, offset=0, max_period=10_000.0)

    assert q_rot[0, 0, 0].tolist() == q[0, 0, 0].tolist()
    assert k_rot[0, 0, 0].tolist() == k[0, 0, 0].tolist()

    cos0 = float(mx.cos(mx.array(1.0, dtype=mx.float32)).item())
    sin0 = float(mx.sin(mx.array(1.0, dtype=mx.float32)).item())
    cos1 = float(mx.cos(mx.array(0.01, dtype=mx.float32)).item())
    sin1 = float(mx.sin(mx.array(0.01, dtype=mx.float32)).item())

    expected_q = [
        (0.5 * cos0) - (-1.0 * sin0),
        (0.5 * sin0) + (-1.0 * cos0),
        (1.5 * cos1) - (-2.0 * sin1),
        (1.5 * sin1) + (-2.0 * cos1),
    ]
    expected_k = [
        (1.25 * cos0) - (0.75 * sin0),
        (1.25 * sin0) + (0.75 * cos0),
        (-0.5 * cos1) - (0.25 * sin1),
        (-0.5 * sin1) + (0.25 * cos1),
    ]

    actual_q = q_rot[0, 0, 1].tolist()
    actual_k = k_rot[0, 0, 1].tolist()
    for expected, actual in zip(expected_q, actual_q):
        assert abs(expected - actual) < 1e-5
    for expected, actual in zip(expected_k, actual_k):
        assert abs(expected - actual) < 1e-5


def test_sanitize_restores_weight_norm_conv_keys() -> None:
    weights = {
        "encoder.0.weight": mx.ones((1,)),
        "decoder.0.weight": mx.ones((2, 2)),
        "quantizer.input_proj.parametrizations.weight.original0": mx.ones((2, 1, 1)),
        "quantizer.input_proj.parametrizations.weight.original1": mx.ones((2, 3, 1)),
        "quantizer.input_proj.bias": mx.zeros((2,)),
    }

    sanitized, skipped, renamed = sanitize_state_dict(weights)

    assert "encoder.0.weight" in sanitized
    assert "quantizer.input_proj.weight" in sanitized
    assert sanitized["quantizer.input_proj.weight"].shape == (2, 3, 1)
    assert skipped == ()
    assert len(renamed) == 2


def test_quantized_codec_round_trip_loads_and_encodes(tmp_path) -> None:
    config = _tiny_codec_config()
    model = MossAudioTokenizerModel(config)
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")

    quantize_moss_audio_tokenizer_model(model, quantization)
    save_moss_audio_tokenizer_model(
        model,
        tmp_path,
        config=config,
        quantization=quantization,
    )

    loaded = load_moss_audio_tokenizer_model(tmp_path, prefer_mlx_int8=False, strict=True)

    assert loaded.quantization == quantization
    assert loaded.alignment_report.is_exact_match
    assert isinstance(loaded.model.encoder[0].transformer.layers[0].linear1, nn.QuantizedLinear)
    assert isinstance(loaded.model.decoder[0].output_proj, nn.QuantizedLinear)
    assert isinstance(loaded.model.decoder[0].transformer.layers[0].linear1, nn.QuantizedLinear)

    wav = mx.linspace(-0.25, 0.25, 12, dtype=mx.float32)
    encoded = loaded.model.encode(wav)
    output = loaded.model.decode(encoded.audio_codes)
    assert encoded.encoder_hidden_states is not None
    assert encoded.encoder_hidden_states.dtype == mx.bfloat16
    assert output.audio.shape == (1, 1, 12)
    assert output.audio.dtype == mx.float32


def test_original_codec_runtime_state_dict_preserves_source_dtype() -> None:
    config = _tiny_codec_config()
    model = MossAudioTokenizerModel(config)
    state_dict = tree_flatten(model.parameters(), destination={})
    bf16_state = {key: value.astype(mx.bfloat16) for key, value in state_dict.items()}

    runtime_state = prepare_runtime_state_dict(bf16_state, quantization=None)

    assert runtime_state["quantizer.quantizers.0.codebook.weight"].dtype == mx.bfloat16
    assert runtime_state["encoder.0.transformer.layers.0.self_attn.in_projs.0.weight"].dtype == mx.bfloat16
    assert (
        runtime_state["decoder.0.transformer.layers.0.self_attn.in_projs.0.weight"].dtype
        == mx.bfloat16
    )
