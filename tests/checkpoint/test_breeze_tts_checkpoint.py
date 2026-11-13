"""The local Breeze packages load strictly in both supported precisions."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_speech.generation.breeze_tts import load_breeze_speech


@pytest.mark.checkpoint
@pytest.mark.parametrize("precision", ["mlx-bf16", "mlx-int8"])
def test_local_breeze_checkpoint_loads_every_component(precision):
    root = Path("models/breezeblue/breeze_tts_2") / precision
    if not (root / "model.safetensors").exists():
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
