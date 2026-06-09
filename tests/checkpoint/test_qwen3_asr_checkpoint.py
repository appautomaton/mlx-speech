"""Header-level Qwen3-ASR MLX checkpoint inspection.

Tier-2 test: requires the local converted Qwen3-ASR MLX package.
The test reads safetensors tensor headers; it does not load full payloads.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from safetensors import safe_open


QWEN_DIR = Path("models/Qwen3-ASR-1.7B-MLX-BF16")

pytestmark = pytest.mark.skipif(
    not (QWEN_DIR / "model.safetensors").exists(),
    reason="Qwen3-ASR MLX checkpoint not present; skipping",
)


def test_qwen3_asr_mlx_checkpoint_namespaces():
    with safe_open(QWEN_DIR / "model.safetensors", framework="numpy") as handle:
        keys = list(handle.keys())
    namespaces = Counter(key.split(".")[0] for key in keys)
    qwen_namespaces = Counter(".".join(key.split(".")[:2]) for key in keys)

    assert len(keys) == 708
    assert namespaces == {"audio_tower": 397, "text_decoder": 311}
    assert qwen_namespaces == {
        "audio_tower.conv2d1": 2,
        "audio_tower.conv2d2": 2,
        "audio_tower.conv2d3": 2,
        "audio_tower.conv_out": 1,
        "audio_tower.layers": 384,
        "audio_tower.ln_post": 2,
        "audio_tower.proj1": 2,
        "audio_tower.proj2": 2,
        "text_decoder.lm_head": 1,
        "text_decoder.model": 310,
    }


def test_qwen3_asr_mlx_checkpoint_header_shapes_and_dtypes():
    with safe_open(QWEN_DIR / "model.safetensors", framework="numpy") as handle:
        expected = {
            "audio_tower.conv2d1.weight": ([480, 3, 3, 1], "BF16"),
            "audio_tower.conv2d2.weight": ([480, 3, 3, 480], "BF16"),
            "audio_tower.conv2d3.weight": ([480, 3, 3, 480], "BF16"),
            "audio_tower.conv_out.weight": ([1024, 7680], "BF16"),
            "text_decoder.model.embed_tokens.weight": ([151936, 2048], "BF16"),
            "text_decoder.lm_head.weight": ([151936, 2048], "BF16"),
        }
        for key, (shape, dtype) in expected.items():
            tensor = handle.get_slice(key)
            assert tensor.get_shape() == shape
            assert tensor.get_dtype() == dtype
