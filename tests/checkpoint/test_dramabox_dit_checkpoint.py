"""Checkpoint loading test for the DramaBox DiT (1457 keys, 3.3B params)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.dit import DiTConfig, LTXModel, load_dit_weights

DIT_PATH = Path("models/dramabox/dramabox-dit-v1.safetensors")

pytestmark = pytest.mark.skipif(
    not DIT_PATH.is_file(),
    reason="DramaBox DiT shard not present",
)


def test_dit_loads_all_keys():
    cfg = DiTConfig()
    model = LTXModel(cfg)
    state = mx.load(str(DIT_PATH))
    n = load_dit_weights(model, state)
    # Expected: 17 top-level + 30 × 48 = 1440 block = 1457
    assert n == 1457

    # Spot-check shapes
    assert model.audio_patchify_proj.weight.shape == (2048, 128)
    assert model.audio_proj_out.weight.shape == (128, 2048)
    assert model.audio_scale_shift_table.shape == (2, 2048)
    # Block 0 spot-check
    blk = model.transformer_blocks[0]
    assert blk.audio_attn1.to_q.weight.shape == (2048, 2048)
    assert blk.audio_attn1.to_gate_logits.weight.shape == (32, 2048)
    assert blk.audio_attn2.to_k.weight.shape == (2048, 2048)
    assert blk.audio_ff.net[0].proj.weight.shape == (8192, 2048)
    assert blk.audio_scale_shift_table.shape == (9, 2048)
    assert blk.audio_prompt_scale_shift_table.shape == (2, 2048)
