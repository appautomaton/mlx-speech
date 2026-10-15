import pytest

from mlx_speech.models.fish_drama_3 import ExpertQuantization, FishDrama3Config


def tiny_payload(*, fp8: bool = True) -> dict:
    """An upstream-shaped Drama 3 config, scaled down."""
    payload = {
        "model_type": "fish_omni",
        "speech_template_version": "reference-bracket-v1",
        "audio_encoder_config": None,
        "eos_token_id": 150,
        "pad_token_id": 151,
        "audio_pad_token_id": 152,
        "semantic_start_token_id": 200,
        "semantic_end_token_id": 263,
        "text_config": {
            "model_type": "fish_omni_text",
            "vocab_size": 300,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "mlp_layer_types": ["moe", "moe"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "rope_parameters": {
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 128,
            "norm_topk_prob": True,
            "attn_output_gate": True,
            "use_qk_norm": True,
            "layernorm_zero_centered_gamma": True,
            "shared_expert_gate": True,
            "rotary_interleaved": False,
            "tie_word_embeddings": False,
        },
        "audio_decoder_config": {
            "decoder_type": "codebook_mtp_shared",
            "dim": 128,
            "text_dim": 128,
            "intermediate_size": 256,
            "n_head": 2,
            "n_local_heads": 1,
            "head_dim": 64,
            "norm_eps": 1e-6,
            "rope_base": 10_000_000.0,
            "rotary_percent": 0.25,
            "rotary_interleaved": True,
            "vocab_size": 64,
            "num_codebooks": 3,
            "n_layer": 2,
            "shared_enrichment_layers": 2,
            "vq_backbone_codebooks": 1,
            "vq_feature_placement": "backbone_partial",
            "attention_output_gate": True,
            "attention_qk_norm": False,
            "layernorm_zero_centered_gamma": True,
            "use_moe": False,
            "mtp_use_swa": False,
            "attention_qkv_bias": False,
            "attention_o_bias": False,
            "dropout": 0.0,
        },
    }
    if fp8:
        payload["quantization_config"] = {
            "quant_method": "bsgemm-moe",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        }
    return payload


def test_upstream_config_marks_fp8_experts():
    config = FishDrama3Config.from_dict(tiny_payload())

    assert config.fp8_experts
    assert config.expert_quantization is None
    assert config.text.layer_types == ("linear_attention", "full_attention")
    assert config.text.rotary_dim == 16
    assert config.audio_decoder.rotary_dim == 16
    assert config.eos_token_id == 150


def test_converted_config_carries_expert_quantization():
    payload = tiny_payload(fp8=False)
    payload["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}

    config = FishDrama3Config.from_dict(payload)

    assert not config.fp8_experts
    assert config.expert_quantization == ExpertQuantization(bits=8, group_size=64)


def test_layer_types_follow_full_attention_interval_when_absent():
    payload = tiny_payload()
    text = payload["text_config"]
    del text["layer_types"]
    text.update(num_hidden_layers=8, mlp_layer_types=["moe"] * 8)
    text["full_attention_interval"] = 4

    config = FishDrama3Config.from_dict(payload)

    assert [
        i for i, t in enumerate(config.text.layer_types) if t == "full_attention"
    ] == [3, 7]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda p: p.update(eos_token_id=210), ValueError, "semantic band"),
        (lambda p: p.update(semantic_end_token_id=262), ValueError, "vocabulary"),
        (
            lambda p: p.update(quantization={"bits": 8, "group_size": 64}),
            ValueError,
            "both",
        ),
        (
            lambda p: p["quantization_config"].update(quant_method="fp8"),
            NotImplementedError,
            "bsgemm-moe",
        ),
        (
            lambda p: p.update(speech_template_version="v0"),
            NotImplementedError,
            "speech_template_version",
        ),
        (
            lambda p: p["text_config"].update(tie_word_embeddings=True),
            NotImplementedError,
            "lm_head",
        ),
        (
            lambda p: p["text_config"].update(num_experts_per_tok=5),
            ValueError,
            "num_experts",
        ),
        (
            lambda p: p["audio_decoder_config"].update(n_layer=3),
            ValueError,
            "shared_enrichment_layers",
        ),
        (
            lambda p: p["audio_decoder_config"].update(use_moe=True),
            NotImplementedError,
            "use_moe",
        ),
    ],
)
def test_rejects_unsupported_configs(mutate, error, match):
    payload = tiny_payload()
    mutate(payload)

    with pytest.raises(error, match=match):
        FishDrama3Config.from_dict(payload)
