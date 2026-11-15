"""Loading the upstream Drama 3 release as is: every dense tensor exactly as released,
and the block-FP8 experts requantized in memory, close to their exact values."""

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import mlx_speech.generation.fish_drama_3 as generation
from mlx_speech.models.fish_drama_3 import ExpertQuantization, FishDrama3Config
from mlx_speech.models.fish_drama_3.checkpoint import (
    EXPERT_PROJECTIONS,
    UPSTREAM_EXPERT_QUANTIZATION,
    upstream_weights,
)
from mlx_speech.models.fish_drama_3.model import (
    FishDrama3Model,
    load_fish_drama_3_model,
)
from tests.unit.test_fish_drama_3_checkpoint import block_oracle, random_fp8_bits
from tests.unit.test_fish_drama_3_config import tiny_payload

# MLX prefix -> upstream prefix; the more specific prefix of backbone comes first.
_UPSTREAM_PREFIX = (
    ("backbone.lm_head.", "lm_head."),
    ("backbone.", "model.language_model."),
    ("audio_decoder.codebook_embeddings.", "audio_decoder.codebook_embeddings."),
    ("audio_decoder.shared_layers.", "audio_decoder_mtp_shared.layers."),
    ("audio_decoder.residual_layers.", "audio_decoder_mtp.layers."),
    ("audio_decoder.codebook_heads.", "audio_decoder_mtp.heads."),
)
_EXPERT_PREFIX = "backbone.layers.{layer}.mlp.experts.{proj}"
_UPSTREAM_EXPERT = "model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"


def _upstream_name(name: str) -> str:
    for mlx_prefix, upstream_prefix in _UPSTREAM_PREFIX:
        if name.startswith(mlx_prefix):
            return upstream_prefix + name[len(mlx_prefix) :]
    raise AssertionError(f"no upstream prefix for {name}")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """BF16 as uint16 bits, F8_E4M3 as uint8 bits, F32 as float32."""
    dtypes = {np.uint16: "BF16", np.uint8: "F8_E4M3", np.float32: "F32"}
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        data = array.tobytes()
        header[name] = {
            "dtype": dtypes[array.dtype.type],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def bf16_bits(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.standard_normal(shape).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def bf16_value(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def write_tiny_release(directory: Path) -> dict[str, np.ndarray]:
    """A complete upstream release of the tiny config in two shards; returns its
    tensors by upstream name."""
    rng = np.random.default_rng(1)
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(tiny_payload()))
    (directory / "codec.pth").write_bytes(b"codec")

    # The loaded model's parameters name every tensor the release must hold.
    shapes = FishDrama3Config.from_dict(
        tiny_payload(fp8=False) | {"quantization": UPSTREAM_EXPERT_QUANTIZATION.to_dict()}
    )
    tensors = {}
    for name, value in tree_flatten(FishDrama3Model(shapes).parameters()):
        if ".mlp.experts." in name:
            continue
        shape = tuple(value.shape)
        if name.endswith(".linear_attn.conv1d.weight"):
            shape = (shape[0], shape[2], shape[1])  # MLX [C, K, 1] -> torch [C, 1, K]
        tensors[_upstream_name(name)] = bf16_bits(rng, shape)
    text = shapes.text
    dims = {
        "gate_proj": (text.moe_intermediate_size, text.hidden_size),
        "up_proj": (text.moe_intermediate_size, text.hidden_size),
        "down_proj": (text.hidden_size, text.moe_intermediate_size),
    }
    for layer in range(text.num_hidden_layers):
        for expert in range(text.num_experts):
            for proj in EXPERT_PROJECTIONS:
                rows, cols = dims[proj]
                prefix = _UPSTREAM_EXPERT.format(layer=layer, expert=expert, proj=proj)
                tensors[f"{prefix}.weight"] = random_fp8_bits(rng, (rows, cols))
                tensors[f"{prefix}.weight_scale_inv"] = rng.uniform(
                    1e-4, 2e-3, (rows // 128, cols // 128)
                ).astype(np.float32)

    names = sorted(tensors)
    weight_map = {}
    for shard, part in (
        ("model-00001-of-00002.safetensors", names[: len(names) // 2]),
        ("model-00002-of-00002.safetensors", names[len(names) // 2 :]),
    ):
        write_safetensors(directory / shard, {n: tensors[n] for n in part})
        weight_map.update(dict.fromkeys(part, shard))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tensors


def test_release_loads_dense_weights_as_released_and_experts_near_fp8(tmp_path):
    release = tmp_path / "drama-3"
    tensors = write_tiny_release(release)

    model = load_fish_drama_3_model(release)

    assert model.config.expert_quantization == UPSTREAM_EXPERT_QUANTIZATION
    assert not model.config.fp8_experts
    params = dict(tree_flatten(model.parameters()))
    for name, value in params.items():
        if ".mlp.experts." in name:
            continue
        assert value.dtype == mx.bfloat16, name
        expected = bf16_value(tensors[_upstream_name(name)])
        if name.endswith(".linear_attn.conv1d.weight"):
            expected = expected.transpose(0, 2, 1)
        np.testing.assert_array_equal(
            np.array(value.astype(mx.float32)), expected, err_msg=name
        )

    text = model.config.text
    for layer in range(text.num_hidden_layers):
        for proj in EXPERT_PROJECTIONS:
            prefix = _EXPERT_PREFIX.format(layer=layer, proj=proj)
            restored = mx.dequantize(
                params[f"{prefix}.weight"],
                params[f"{prefix}.scales"],
                params[f"{prefix}.biases"],
                group_size=UPSTREAM_EXPERT_QUANTIZATION.group_size,
                bits=UPSTREAM_EXPERT_QUANTIZATION.bits,
            )
            exact = np.stack(
                [
                    block_oracle(
                        tensors[f"{source}.weight"], tensors[f"{source}.weight_scale_inv"]
                    )
                    for source in (
                        _UPSTREAM_EXPERT.format(layer=layer, expert=e, proj=proj)
                        for e in range(text.num_experts)
                    )
                ]
            )
            got = np.array(restored.astype(mx.float32), dtype=np.float64)
            assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-2, prefix


def test_upstream_reader_refuses_a_release_missing_an_expert_tensor(tmp_path):
    release = tmp_path / "drama-3"
    write_tiny_release(release)
    index_path = release / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"][
        _UPSTREAM_EXPERT.format(layer=1, expert=3, proj="down_proj") + ".weight_scale_inv"
    ]
    index_path.write_text(json.dumps(index))
    config = FishDrama3Config.from_path(release)

    with pytest.raises(ValueError, match="1 missing"):
        next(upstream_weights(release, config, UPSTREAM_EXPERT_QUANTIZATION))


def test_upstream_reader_checks_the_mode_before_reading_any_tensor(tmp_path):
    release = tmp_path / "drama-3"
    write_tiny_release(release)
    config = FishDrama3Config.from_path(release)
    weights = upstream_weights(
        release, config, ExpertQuantization(bits=4, group_size=32, mode="mxfp4")
    )
    with pytest.raises(NotImplementedError, match="affine"):
        next(weights)


def test_runtime_builds_the_codec_from_codec_pth(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        generation.FishS2Codec,
        "from_codec_pth",
        lambda path: calls.append(Path(path)) or "codec",
    )
    monkeypatch.setattr(
        generation, "load_fish_drama_3_model", lambda path, dense_int8: "model"
    )
    monkeypatch.setattr(generation.Tokenizer, "from_file", lambda path: "tokenizer")

    runtime = generation.FishDrama3Runtime.from_dir(tmp_path)

    assert calls == [tmp_path / "codec.pth"]
    assert runtime.codec == "codec"


@pytest.mark.parametrize(("bits", "group_size"), [(8, 64), (5, 64), (4, 32)])
def test_mlx_format_weights_load_as_their_config_records(tmp_path, bits, group_size):
    """Whatever quantization the weights record, they load as stored."""
    quantization = {"bits": bits, "group_size": group_size, "mode": "affine"}
    config = FishDrama3Config.from_dict(
        tiny_payload(fp8=False) | {"quantization": quantization}
    )
    mx.random.seed(bits)
    stored = FishDrama3Model(config)
    for proj in EXPERT_PROJECTIONS:  # real packed experts, not the zero placeholders
        linear = getattr(stored.backbone.layers[0].mlp.experts, proj)
        dense = mx.random.normal(
            (linear.weight.shape[0], linear.weight.shape[1], linear.scales.shape[2] * group_size)
        )
        linear.weight, linear.scales, linear.biases = mx.quantize(
            dense, group_size=group_size, bits=bits
        )
    weights = dict(tree_flatten(stored.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(tiny_payload(fp8=False) | {"quantization": quantization})
    )

    loaded = load_fish_drama_3_model(tmp_path)

    assert loaded.config.expert_quantization == ExpertQuantization(bits, group_size)
    for name, value in tree_flatten(loaded.parameters()):
        assert mx.array_equal(value, weights[name]).item(), name


def test_loader_refuses_weights_that_are_neither_release_nor_mlx(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(tiny_payload(fp8=False)))

    with pytest.raises(ValueError, match="neither"):
        load_fish_drama_3_model(tmp_path)


@pytest.mark.parametrize("bits", [8, 4])
def test_min_max_quantization_uses_mlx_layout_exactly(bits):
    """Values on a known grid come back exactly through mx.dequantize."""
    from mlx_speech.models.fish_drama_3.checkpoint import quantize_min_max

    rng = np.random.default_rng(bits)
    levels = 2**bits - 1
    q = rng.integers(0, levels + 1, size=(2, 8, 2, 64))
    q[..., 0], q[..., 1] = 0, levels  # every group spans its full range
    scale = mx.array(rng.uniform(1e-4, 1e-2, (2, 8, 2, 1)).astype(np.float32))
    bias = mx.array(rng.uniform(-1, 0, (2, 8, 2, 1)).astype(np.float32))
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    bias = bias.astype(mx.bfloat16).astype(mx.float32)
    w = (mx.array(q.astype(np.float32)) * scale + bias).reshape(2, 8, 128)

    packed, scales, biases = quantize_min_max(w, 64, bits)
    assert packed.dtype == mx.uint32 and packed.shape == (2, 8, 128 * bits // 32)
    assert scales.dtype == biases.dtype == mx.bfloat16
    restored = mx.dequantize(
        packed,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        group_size=64,
        bits=bits,
    )
    np.testing.assert_allclose(np.array(restored), np.array(w), rtol=0, atol=1e-6)


def test_min_max_quantization_restores_a_constant_group():
    from mlx_speech.models.fish_drama_3.checkpoint import quantize_min_max

    w = mx.full((1, 64), 0.375, dtype=mx.bfloat16)
    packed, scales, biases = quantize_min_max(w, 64, 8)
    restored = mx.dequantize(packed, scales, biases, group_size=64, bits=8)
    assert mx.array_equal(restored, w).item()


def test_min_max_quantization_beats_mx_quantize_on_fp8_experts():
    from mlx_speech.models.fish_drama_3.checkpoint import (
        dequantize_fp8_blocks,
        quantize_min_max,
    )

    rng = np.random.default_rng(3)
    bits = random_fp8_bits(rng, (4, 256, 256))
    scale = rng.uniform(1e-4, 2e-3, (4, 2, 2)).astype(np.float32)
    exact = np.stack([block_oracle(bits[e], scale[e]) for e in range(4)])
    w = dequantize_fp8_blocks(mx.array(bits), mx.array(scale)).astype(mx.bfloat16)

    def error(packed, scales, biases):
        got = mx.dequantize(packed, scales, biases, group_size=64, bits=8)
        got = np.array(got.astype(mx.float32), dtype=np.float64)
        return np.linalg.norm(got - exact) / np.linalg.norm(exact)

    ours = error(*quantize_min_max(w, 64, 8))
    builtin = error(*mx.quantize(w, group_size=64, bits=8))
    assert ours < builtin
    assert ours < 1e-2
