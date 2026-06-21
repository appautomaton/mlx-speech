"""Checkpoint loading test for the RE-USE / SEMamba MLX port.

Structural gate (Slice 4): every torch key maps onto an MLX `SEMamba` param
with none missing and none extra, and a forward on a dummy STFT-shaped input
returns correctly-shaped ``(amp_g, pha_g)``. Numeric parity is Slice 6.

Skips cleanly when the local checkpoint is absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from safetensors import safe_open

from mlx_speech.models.reuse.loader import (
    assert_keys_match,
    build_mlx_state,
    load_mlx_semamba,
    load_semamba,
    torch_key_to_mlx,
)
from mlx_speech.models.reuse.semamba import SEMamba

CHECKPOINT = Path("models/reuse/original")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(),
    reason="RE-USE checkpoint not present",
)

# n_fft=320 -> 161 freq bins; small T keeps the forward fast.
_F = 161
_T = 12


def test_converted_keys_match_model_exactly():
    state = build_mlx_state(CHECKPOINT)
    model_keys = {k for k, _ in tree_flatten(SEMamba().parameters())}

    # Shared strict check: raises with a sampled diff if the sets disagree.
    assert_keys_match(model_keys, set(state))
    # The reference checkpoint has 1416 keys / ~9.61M params.
    assert len(state) == 1416


def test_every_torch_key_maps_to_a_model_param():
    model_keys = {k for k, _ in tree_flatten(SEMamba().parameters())}
    with safe_open(str(CHECKPOINT / "model.safetensors"), "numpy") as f:
        torch_keys = list(f.keys())
    assert len(torch_keys) == 1416
    for key in torch_keys:
        assert torch_key_to_mlx(key) in model_keys, key


def test_conv_weight_layout_is_remapped():
    state = build_mlx_state(CHECKPOINT)
    # Conv2d torch [out, in, kh, kw] -> MLX [out, kh, kw, in].
    # dense_conv_1: torch (64, 2, 1, 1) -> MLX (64, 1, 1, 2).
    assert state["dense_encoder.dense_conv_1.conv.weight"].shape == (64, 1, 1, 2)
    # Conv1d torch [out, in/g, kw] -> MLX [out, kw, in/g].
    # mamba conv1d: torch (256, 1, 4) -> MLX (256, 4, 1).
    assert state["TSMamba.0.time_mamba.forward_blocks.conv1d.weight"].shape == (
        256,
        4,
        1,
    )


def test_strict_load_and_forward_shapes():
    model = load_semamba(CHECKPOINT, strict=True)
    mag = mx.random.normal((1, _F, _T)) * 0.1
    pha = mx.random.normal((1, _F, _T)) * 0.1
    amp_g, pha_g, com_g = model(mag, pha)
    mx.eval(amp_g, pha_g, com_g)

    assert amp_g.shape == (1, _F, _T)
    assert pha_g.shape == (1, _F, _T)
    assert com_g.shape == (1, _F, _T, 2)
    assert bool(mx.all(mx.isfinite(amp_g)).item())
    assert bool(mx.all(mx.isfinite(pha_g)).item())


def test_conversion_roundtrips_through_saved_file(tmp_path: Path):
    from scripts.convert.reuse import convert

    summary = convert(CHECKPOINT, tmp_path)
    assert summary["num_keys"] == 1416
    assert (tmp_path / "model.safetensors").is_file()

    model = load_mlx_semamba(tmp_path)
    amp_g, pha_g, _ = model(mx.zeros((1, _F, _T)), mx.zeros((1, _F, _T)))
    mx.eval(amp_g, pha_g)
    assert amp_g.shape == (1, _F, _T)
    assert pha_g.shape == (1, _F, _T)
