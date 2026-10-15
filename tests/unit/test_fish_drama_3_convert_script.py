import json
import struct

import mlx.core as mx
import numpy as np
import pytest

import scripts.convert.fish_drama_3 as convert_script
from mlx_speech.checkpoints.sharded import load_state_dict
from mlx_speech.models.fish_drama_3 import ExpertQuantization, FishDrama3Config
from tests.unit.test_fish_drama_3_checkpoint import block_oracle, random_fp8_bits
from tests.unit.test_fish_drama_3_config import tiny_payload

QUANT = ExpertQuantization(bits=8, group_size=64)
# Upstream name -> (MLX name, shape); one tensor per remapped prefix.
DENSE = {
    "model.language_model.embed_tokens.weight": (
        "backbone.embed_tokens.weight",
        (300, 128),
    ),
    "model.language_model.layers.0.linear_attn.conv1d.weight": (
        "backbone.layers.0.linear_attn.conv1d.weight",
        (256, 1, 4),
    ),
    "model.language_model.layers.1.self_attn.q_proj.weight": (
        "backbone.layers.1.self_attn.q_proj.weight",
        (256, 128),
    ),
    "model.language_model.norm.weight": ("backbone.norm.weight", (128,)),
    "lm_head.weight": ("backbone.lm_head.weight", (300, 128)),
    "audio_decoder.codebook_embeddings.weight": (
        "audio_decoder.codebook_embeddings.weight",
        (192, 128),
    ),
    "audio_decoder_mtp_shared.layers.0.eh_proj.weight": (
        "audio_decoder.shared_layers.0.eh_proj.weight",
        (128, 256),
    ),
    "audio_decoder_mtp.layers.1.codebook_embedding.weight": (
        "audio_decoder.residual_layers.1.codebook_embedding.weight",
        (64, 128),
    ),
    "audio_decoder_mtp.heads.0.weight": (
        "audio_decoder.codebook_heads.0.weight",
        (64, 128),
    ),
}
# moe_intermediate_size 256, hidden 128: gate/up [256, 128], down [128, 256].
EXPERT_SHAPES = {
    "gate_proj": (256, 128),
    "up_proj": (256, 128),
    "down_proj": (128, 256),
}
SUPPORTING = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
)


def write_safetensors(path, tensors):
    """tensors: name -> (safetensors dtype, shape, raw bytes); F8_E4M3 included."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def bf16_bits(rng, shape):
    values = rng.standard_normal(shape).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def bf16_value(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture
def upstream(tmp_path):
    """A two-layer upstream release whose layer 0 experts straddle both shards."""
    rng = np.random.default_rng(0)
    source = tmp_path / "drama-3"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(tiny_payload()))
    (source / "codec.pth").write_bytes(b"codec")
    for name in SUPPORTING:
        (source / name).write_text(name)

    dense = {name: bf16_bits(rng, shape) for name, (_, shape) in DENSE.items()}
    experts = {}
    for layer in range(2):
        for expert in range(4):
            for proj, shape in EXPERT_SHAPES.items():
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                )
                experts[f"{prefix}.weight"] = random_fp8_bits(rng, shape)
                experts[f"{prefix}.weight_scale_inv"] = rng.uniform(
                    1e-4, 2e-3, size=(shape[0] // 128, shape[1] // 128)
                ).astype(np.float32)

    def entry(array):
        dtype = {np.uint16: "BF16", np.uint8: "F8_E4M3", np.float32: "F32"}[
            array.dtype.type
        ]
        return dtype, array.shape, array.tobytes()

    names = sorted(experts)
    first = {**dense, **{n: experts[n] for n in names[:5]}}
    second = {n: experts[n] for n in names[5:]}
    weight_map = {}
    for shard, tensors in (
        ("model-00001-of-00002.safetensors", first),
        ("model-00002-of-00002.safetensors", second),
    ):
        write_safetensors(source / shard, {n: entry(a) for n, a in tensors.items()})
        weight_map.update(dict.fromkeys(tensors, shard))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return source, dense, experts


@pytest.fixture
def codec_calls(monkeypatch):
    calls = []

    def fake_codec(codec_pth, output_dir):
        output_dir.mkdir(parents=True)
        (output_dir / "config.json").write_text("{}")
        calls.append((codec_pth, output_dir))

    monkeypatch.setattr(convert_script, "convert_codec_pth_to_assets", fake_codec)
    return calls


def test_converts_upstream_release_into_mlx_checkpoint(upstream, codec_calls, tmp_path):
    source, dense, experts = upstream
    output = tmp_path / "mlx-int8"

    convert_script.convert_fish_drama_3(
        source, output, quantization=QUANT, shard_bytes=200_000
    )

    loaded = load_state_dict(output)
    assert len(loaded.files) > 1
    expert_names = {
        f"backbone.layers.{layer}.mlp.experts.{proj}.{part}"
        for layer in range(2)
        for proj in EXPERT_SHAPES
        for part in ("weight", "scales", "biases")
    }
    assert set(loaded.weights) == {name for name, _ in DENSE.values()} | expert_names

    for upstream_name, (name, _) in DENSE.items():
        value = loaded.weights[name]
        assert value.dtype == mx.bfloat16
        expected = bf16_value(dense[upstream_name])
        if name.endswith("conv1d.weight"):
            expected = expected.transpose(0, 2, 1)
        np.testing.assert_array_equal(np.array(value.astype(mx.float32)), expected)

    for layer in range(2):
        for proj, (rows, cols) in EXPERT_SHAPES.items():
            prefix = f"backbone.layers.{layer}.mlp.experts.{proj}"
            weight = loaded.weights[f"{prefix}.weight"]
            assert weight.shape == (4, rows, cols * QUANT.bits // 32)
            restored = mx.dequantize(
                weight,
                loaded.weights[f"{prefix}.scales"],
                loaded.weights[f"{prefix}.biases"],
                group_size=QUANT.group_size,
                bits=QUANT.bits,
            )
            base = f"model.language_model.layers.{layer}.mlp.experts"
            exact = np.stack(
                [
                    block_oracle(
                        experts[f"{base}.{e}.{proj}.weight"],
                        experts[f"{base}.{e}.{proj}.weight_scale_inv"],
                    )
                    for e in range(4)
                ]
            )
            got = np.array(restored.astype(mx.float32), dtype=np.float64)
            assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-2

    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config
    assert config["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}
    assert FishDrama3Config.from_path(output).expert_quantization == QUANT
    assert codec_calls == [(source / "codec.pth", output / "codec-mlx")]
    for name in SUPPORTING:
        assert (output / name).read_text() == name


def test_refuses_a_release_missing_an_expert_tensor(upstream, codec_calls, tmp_path):
    source, _, _ = upstream
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"][
        "model.language_model.layers.1.mlp.experts.3.down_proj.weight_scale_inv"
    ]
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="1 missing"):
        convert_script.convert_fish_drama_3(
            source, tmp_path / "out", quantization=QUANT
        )
    assert codec_calls == []


def test_refuses_a_checkpoint_without_fp8_experts(tmp_path, codec_calls):
    (tmp_path / "config.json").write_text(json.dumps(tiny_payload(fp8=False)))

    with pytest.raises(ValueError, match="block-FP8"):
        convert_script.convert_fish_drama_3(
            tmp_path, tmp_path / "out", quantization=QUANT
        )
