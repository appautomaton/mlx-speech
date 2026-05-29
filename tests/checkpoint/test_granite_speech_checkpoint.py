"""Header-level Granite Speech checkpoint inspection.

Tier-2 test: requires the local Granite Speech original checkpoint directory.
The test reads the safetensors index only; it does not load tensor payloads.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mlx_speech.checkpoints.sharded import ShardedCheckpointIndex


GRANITE_DIR = Path("models/ibm/granite_4_0_1b_speech/original")

pytestmark = pytest.mark.skipif(
    not (GRANITE_DIR / "model.safetensors.index.json").exists(),
    reason="Granite Speech checkpoint index not present; skipping",
)


def test_granite_original_checkpoint_index_namespaces():
    index = ShardedCheckpointIndex.from_directory(GRANITE_DIR)
    namespaces = Counter(key.split(".")[0] for key in index.weight_map)

    assert len(index.weight_map) == 954
    assert namespaces == {
        "language_model": 363,
        "encoder": 534,
        "projector": 57,
    }
    assert len(index.shard_paths) == 3
    assert all(path.exists() for path in index.shard_paths)
