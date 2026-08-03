"""Full Granite Speech checkpoint load.

Tier-2 test: requires the local Granite Speech original checkpoint directory.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_speech.models.granite_speech_asr import (
    GraniteSpeechModel,
    QuantizationConfig,
    get_quantization_config,
)


GRANITE_DIR = Path("models/ibm/granite_4_0_1b_speech/original")
GRANITE_INT8_DIR = Path("models/ibm/granite_4_0_1b_speech/mlx-int8")


@pytest.mark.skipif(
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


@pytest.mark.skipif(
    not (GRANITE_INT8_DIR / "model.safetensors").exists(),
    reason="Granite Speech int8 artifact not present; skipping",
)
def test_granite_speech_int8_full_model_strict_loads_saved_module_tree():
    loaded = GraniteSpeechModel.from_dir(
        GRANITE_INT8_DIR,
        dtype=mx.bfloat16,
        strict=True,
    )

    assert loaded.alignment.is_exact_match
    assert len(loaded.source_files) == 1
    assert loaded.skipped_keys == ()
    assert loaded.transposed_keys == ()
    assert get_quantization_config(loaded.config) == QuantizationConfig()
    assert isinstance(
        loaded.model.language_model.model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(loaded.model.encoder.input_linear, nn.Linear)
    assert isinstance(loaded.model.projector.linear, nn.Linear)
