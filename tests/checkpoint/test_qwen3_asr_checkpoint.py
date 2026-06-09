"""Header-level Qwen3-ASR checkpoint inspection.

Tier-2 test: requires the local Qwen3-ASR original checkpoint directory.
The test reads the safetensors index and tensor headers; it does not load full payloads.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from safetensors import safe_open

from mlx_speech.checkpoints.sharded import ShardedCheckpointIndex


QWEN_DIR = Path("models/qwen3_asr_1_7b/original")

pytestmark = pytest.mark.skipif(
    not (QWEN_DIR / "model.safetensors.index.json").exists(),
    reason="Qwen3-ASR checkpoint index not present; skipping",
)


def test_qwen3_asr_original_checkpoint_index_namespaces():
    index = ShardedCheckpointIndex.from_directory(QWEN_DIR)
    namespaces = Counter(key.split(".")[0] for key in index.weight_map)
    thinker_namespaces = Counter(".".join(key.split(".")[:2]) for key in index.weight_map)

    assert len(index.weight_map) == 708
    assert namespaces == {"thinker": 708}
    assert thinker_namespaces == {
        "thinker.audio_tower": 397,
        "thinker.model": 310,
        "thinker.lm_head": 1,
    }
    assert len(index.shard_paths) == 2
    assert all(path.exists() for path in index.shard_paths)


def test_qwen3_asr_original_checkpoint_header_shapes_and_dtypes():
    shard = QWEN_DIR / "model-00001-of-00002.safetensors"
    with safe_open(shard, framework="numpy") as handle:
        expected = {
            "thinker.audio_tower.conv2d1.weight": ([480, 1, 3, 3], "BF16"),
            "thinker.audio_tower.conv2d2.weight": ([480, 480, 3, 3], "BF16"),
            "thinker.audio_tower.conv2d3.weight": ([480, 480, 3, 3], "BF16"),
            "thinker.audio_tower.conv_out.weight": ([1024, 7680], "BF16"),
            "thinker.model.embed_tokens.weight": ([151936, 2048], "BF16"),
            "thinker.lm_head.weight": ([151936, 2048], "BF16"),
        }
        for key, (shape, dtype) in expected.items():
            tensor = handle.get_slice(key)
            assert tensor.get_shape() == shape
            assert tensor.get_dtype() == dtype
