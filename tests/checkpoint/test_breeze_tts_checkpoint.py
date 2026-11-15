"""The upstream Breeze snapshot and the local MLX packages load strictly."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_speech.generation.breeze_tts import load_breeze_speech


@pytest.mark.checkpoint
@pytest.mark.parametrize("precision", ["original", "mlx-bf16", "mlx-int8"])
def test_local_breeze_checkpoint_loads_every_component(precision):
    root = Path("models/breezeblue/breeze_tts_2") / precision
    if not (root / "config.json").exists():
        pytest.skip(f"Local Breeze {precision} checkpoint is missing")
    speech, codec, tokenizer = load_breeze_speech(root)
    projection = speech.backbone_model.layers[0].mlp.up_proj
    assert isinstance(
        projection, nn.QuantizedLinear if precision == "mlx-int8" else nn.Linear
    )
    assert speech.config.num_codebooks == 16
    assert speech.lm_head.weight.dtype == mx.bfloat16
    assert codec.decoder.pre_conv.conv.weight.dtype == mx.float32
    assert tokenizer.token_to_id("<bos>") is not None


@pytest.mark.checkpoint
def test_the_upstream_snapshot_loads_exactly_as_its_converted_package():
    root = Path("models/breezeblue/breeze_tts_2")
    if not (root / "original" / "config.json").exists() or not (
        root / "mlx-bf16" / "model.safetensors"
    ).exists():
        pytest.skip("Local Breeze snapshot or BF16 package is missing")

    def tensors(name):
        speech, codec, _ = load_breeze_speech(root / name)
        return {
            **{"speech." + k: v for k, v in tree_flatten(speech.parameters())},
            **{"codec." + k: v for k, v in tree_flatten(codec.parameters())},
        }

    original, package = tensors("original"), tensors("mlx-bf16")
    assert original.keys() == package.keys()
    for key, value in original.items():
        assert value.dtype == package[key].dtype, key
        assert mx.array_equal(value, package[key]), key
