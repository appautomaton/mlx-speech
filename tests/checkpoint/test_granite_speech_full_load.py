"""Full Granite Speech checkpoint load.

Tier-2 test: requires the local Granite Speech original checkpoint directory.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.granite_speech_asr import GraniteSpeechModel


GRANITE_DIR = Path("models/ibm/granite_4_0_1b_speech/original")

pytestmark = pytest.mark.skipif(
    not (GRANITE_DIR / "model.safetensors.index.json").exists(),
    reason="Granite Speech checkpoint index not present; skipping",
)


def test_granite_speech_full_model_strict_loads_without_retaining_state_dict():
    loaded = GraniteSpeechModel.from_dir(GRANITE_DIR, dtype=mx.bfloat16, strict=True)

    assert loaded.alignment.is_exact_match
    assert len(loaded.source_files) == 3
    assert len(loaded.skipped_keys) == 16
    assert len(loaded.transposed_keys) == 48
    assert loaded.config.model_type == "granite_speech"
    assert loaded.tokenizer.audio_token_id == loaded.config.audio_token_index
    assert loaded.feature_extractor.sample_rate == 16000
    assert not hasattr(loaded, "checkpoint")
    assert not hasattr(loaded, "state_dict")
