"""FireRedTTS3 read straight from the upstream FP32 release, at upstream precision:
BF16 where upstream autocasts (backbone and RedAE encoder linears), FP32 elsewhere."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import mlx_speech.models.fireredtts3.checkpoint as checkpoint
from mlx_speech.models.fireredtts3.checkpoint import upstream_dtype
from mlx_speech.models.fireredtts3.core import FireRedTTS3Core, FireRedTTS3CoreConfig
from mlx_speech.models.fireredtts3.redae import RedAE, RedAEConfig
from mlx_speech.models.qwen3_asr.text_decoder import NATIVE_DECODE_COMPUTE
from mlx_speech.tts._adapters.fireredtts3 import FireRedTTS3Adapter
from mlx_speech.tts._registry import _resolve_tts_family

TINY_QWEN = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "vocab_size": 32,
    "max_position_embeddings": 64,
    "rope_theta": 10_000,
    "rms_norm_eps": 1e-6,
}
TINY_CORE = {
    "redae_dim": 64,
    "num_history_patches": 2,
    "spk_in_dim": 6,
    "patch_size": 4,
    "patch_encoder_hidden_size": 8,
    "patch_encoder_mlp_ratio": 2,
    "patch_encoder_depth": 1,
    "patch_encoder_num_heads": 2,
    "dit_mlp_ratio": 2,
    "dit_depth": 1,
    "dit_num_heads": 2,
    "dit_hidden_size": 8,
    "dtype": "float32",
}
TINY_REDAE = {
    "audio_patch_size": 4,
    "audio_sample_rate": 24_000,
    "bottleneck_dim": 64,
    "dtype": "float32",
    **{
        f"{side}_{key}": value
        for side in ("enc", "dec")
        for key, value in {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "max_position_embeddings": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "sliding_window": 4,
            "max_window_layers": 0,
        }.items()
    },
    "enc_extra_downsample_rate": 2,
    "enc_downsample_num_hidden_layers": 1,
}


def write_tiny_release(root: Path, monkeypatch) -> dict[str, dict[str, mx.array]]:
    """A tiny upstream release: FP32 tensors in upstream names and layouts."""
    monkeypatch.setattr(checkpoint, "_QWEN_CONFIG", TINY_QWEN)
    for folder, config in (("fireredtts3_base", TINY_CORE), ("redae", TINY_REDAE)):
        (root / folder).mkdir(parents=True)
        (root / folder / "config.json").write_text(json.dumps(config))
    artifact = checkpoint.build_artifact_config(root)

    mx.random.seed(0)
    core = FireRedTTS3Core(FireRedTTS3CoreConfig.from_artifact(artifact))
    redae = RedAE(RedAEConfig.from_artifact(artifact))
    sources = {}
    for component, model, folder in (
        ("core", core, "fireredtts3_base"),
        ("redae", redae, "redae"),
    ):
        tensors = {}
        for name, value in tree_flatten(model.parameters()):
            if ".conv.block." in name and name.endswith(".weight"):
                value = value.transpose(0, 2, 1)  # PyTorch Conv1d layout
            tensors[name] = (value + 1e-3 * mx.random.normal(value.shape)).astype(
                mx.float32
            )
        mx.save_safetensors(str(root / folder / "model.safetensors"), tensors)
        monkeypatch.setitem(checkpoint.EXPECTED_SOURCE_TENSORS, component, len(tensors))
        sources[component] = tensors
    return sources


@pytest.mark.parametrize(
    ("component", "name", "dtype"),
    [
        ("core", "backbone_llm.layers.3.self_attn.q_proj.weight", mx.bfloat16),
        ("core", "backbone_llm.layers.3.mlp.down_proj.weight", mx.bfloat16),
        ("core", "backbone_llm.layers.3.self_attn.q_norm.weight", mx.float32),
        ("core", "backbone_llm.embed_tokens.weight", mx.float32),
        ("core", "patch_encoder.blocks.0.attn.qkv.weight", mx.float32),
        ("core", "dit_head.weight", mx.float32),
        ("core", "dit.blocks.0.attn.qkv.weight", mx.float32),
        ("redae", "encoder.in_proj.0.bias", mx.bfloat16),
        ("redae", "encoder.out_proj.weight", mx.bfloat16),
        ("redae", "encoder.qwen3.layers.0.mlp.up_proj.weight", mx.bfloat16),
        ("redae", "encoder.downsample.qwen3.layers.0.self_attn.o_proj.weight", mx.bfloat16),
        ("redae", "encoder.qwen3.layers.0.input_layernorm.weight", mx.float32),
        ("redae", "encoder.downsample.cls_tok", mx.float32),
        ("redae", "decoder.qwen3.layers.0.self_attn.q_proj.weight", mx.float32),
        ("redae", "decoder.in_proj.weight", mx.float32),
        ("speaker", "head.conv1.weight", mx.float32),
    ],
)
def test_upstream_precision_is_bf16_only_where_upstream_autocasts(component, name, dtype):
    assert upstream_dtype(component, name) == dtype


def test_core_loads_at_upstream_precision_and_computes_its_backbone_in_bf16(
    tmp_path, monkeypatch
):
    source = write_tiny_release(tmp_path, monkeypatch)["core"]

    core = FireRedTTS3Core.from_upstream(tmp_path)

    loaded = dict(tree_flatten(core.parameters()))
    assert loaded.keys() == source.keys()
    for name, value in loaded.items():
        expected = source[name]
        if ".conv.block." in name and name.endswith(".weight"):
            expected = expected.transpose(0, 2, 1)
        assert value.dtype == upstream_dtype("core", name)
        np.testing.assert_array_equal(
            np.array(value.astype(mx.float32)),
            np.array(expected.astype(value.dtype).astype(mx.float32)),
        )
    assert any(v.dtype == mx.bfloat16 for v in loaded.values())
    assert core.backbone_compute == NATIVE_DECODE_COMPUTE


def test_redae_loads_at_upstream_precision_without_unused_token_embeddings(
    tmp_path, monkeypatch
):
    source = write_tiny_release(tmp_path, monkeypatch)["redae"]

    redae = RedAE.from_upstream(tmp_path)

    loaded = dict(tree_flatten(redae.parameters()))
    assert loaded.keys() == {k for k in source if "embed_tokens" not in k}
    for name, value in loaded.items():
        assert value.dtype == upstream_dtype("redae", name)
    assert loaded["encoder.qwen3.layers.0.self_attn.q_proj.weight"].dtype == mx.bfloat16
    assert loaded["decoder.qwen3.layers.0.self_attn.q_proj.weight"].dtype == mx.float32
    audio = mx.linspace(-0.1, 0.1, 32)[None]
    assert redae.decode(redae.encode(audio)).shape[-1] == 32


def test_core_passes_its_backbone_compute_to_every_step(monkeypatch):
    from tests.unit.test_fireredtts3_core import _tiny_config

    mx.random.seed(1)
    core = FireRedTTS3Core(_tiny_config())
    seen = []
    for method in ("prefill", "decode_step"):
        original = getattr(core.backbone_llm, method)
        monkeypatch.setattr(
            core.backbone_llm,
            method,
            lambda original=original, **kwargs: seen.append(kwargs["compute"])
            or original(**kwargs),
        )

    def run():
        core.generate(
            speaker_embedding=mx.zeros((1, 6)),
            text_tokens=mx.array([[1, 2]]),
            prompt_latents=mx.zeros((1, 2, 4)),
            flow_steps=1,
            guidance_scale=0.0,
            stop_threshold=2.0,
            min_generated_patches=0,
            max_generated_patches=2,
        )

    run()
    assert seen == [None, None]  # an MLX artifact keeps the reference path
    seen.clear()
    core.backbone_compute = NATIVE_DECODE_COMPUTE
    run()
    assert seen == [NATIVE_DECODE_COMPUTE] * 2


def test_the_upstream_release_is_found_and_routed(tmp_path, monkeypatch):
    (tmp_path / "fireredtts3_base").mkdir()
    (tmp_path / "fireredtts3_base" / "config.json").write_text("{}")
    assert _resolve_tts_family(tmp_path) == "fireredtts3"
    assert checkpoint.is_upstream_release(tmp_path)

    monkeypatch.setattr(
        FireRedTTS3Adapter, "from_upstream", classmethod(lambda cls, root: ("upstream", root))
    )
    assert FireRedTTS3Adapter.from_dir(tmp_path) == ("upstream", tmp_path)

    (tmp_path / "config.json").write_text('{"model_type": "fireredtts3_base"}')
    assert not checkpoint.is_upstream_release(tmp_path)  # an MLX artifact wins


def test_an_incomplete_upstream_release_names_what_is_missing(tmp_path):
    (tmp_path / "fireredtts3_base").mkdir()
    (tmp_path / "fireredtts3_base" / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="redae/model.safetensors"):
        FireRedTTS3Adapter.from_dir(tmp_path)
