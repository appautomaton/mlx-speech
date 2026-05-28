"""Checkpoint-loading + end-to-end shape test for the DramaBox prompt pipeline.

Tier-2 test: requires `models/dramabox/dramabox-audio-components.safetensors`
and `models/gemma_3_12b_it_4bit/` to be present. Skipped automatically if
either is absent.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.prompt import (
    DramaBoxPromptEncoder,
    Embeddings1DConnector,
    EmbeddingsProcessor,
    FeatureExtractorV2,
    load_audio_components_state,
    load_connector_weights,
    load_feature_extractor_weights,
)
from mlx_speech.models.gemma3_text import LTXVGemmaTokenizer, load_gemma3_text_model

AUDIO_COMPONENTS = Path("models/dramabox/dramabox-audio-components.safetensors")
GEMMA_DIR = Path("models/gemma_3_12b_it_4bit")

pytestmark = pytest.mark.skipif(
    not AUDIO_COMPONENTS.is_file() or not GEMMA_DIR.is_dir(),
    reason="DramaBox audio-components or Gemma 4-bit checkpoint not present",
)


def test_feature_extractor_aggregate_loads_from_audio_components():
    state = load_audio_components_state(AUDIO_COMPONENTS)

    fx = FeatureExtractorV2(embedding_dim=3840, out_features=2048, num_layers=49)
    n_loaded, _ = load_feature_extractor_weights(fx, state)
    assert n_loaded == 2  # weight + bias
    # Shapes
    assert fx.audio_aggregate_embed.weight.shape == (2048, 188160)
    assert fx.audio_aggregate_embed.bias.shape == (2048,)


def test_connector_weights_load_from_audio_components():
    state = load_audio_components_state(AUDIO_COMPONENTS)

    conn = Embeddings1DConnector(
        num_attention_heads=32,
        attention_head_dim=64,
        num_layers=8,
        num_learnable_registers=128,
        positional_embedding_max_pos=4096,
        seq_len=1024,
    )
    n_loaded = load_connector_weights(conn, state)
    # Expected: 1 (learnable_registers) + 8 blocks × 16 keys = 129
    assert n_loaded == 129

    # Spot-check a few specific weights to make sure remap is correct
    assert conn.learnable_registers.shape == (128, 2048)
    block0 = conn.transformer_1d_blocks[0]
    assert block0.attn1.to_q.weight.shape == (2048, 2048)
    assert block0.attn1.to_q.bias.shape == (2048,)
    assert block0.attn1.to_gate_logits.weight.shape == (32, 2048)
    assert block0.attn1.to_gate_logits.bias.shape == (32,)
    # ff.net.0.proj
    assert block0.ff.net[0].proj.weight.shape == (8192, 2048)
    assert block0.ff.net[0].proj.bias.shape == (8192,)
    # ff.net.2
    assert block0.ff.net[2].weight.shape == (2048, 8192)
    assert block0.ff.net[2].bias.shape == (2048,)


def test_prompt_pipeline_end_to_end_shape():
    """Full pipeline: text → Gemma → FeatureExtractorV2 → Connector → a_ctx."""
    state = load_audio_components_state(AUDIO_COMPONENTS)

    # Load Gemma
    gemma, _ = load_gemma3_text_model(GEMMA_DIR)
    tokenizer = LTXVGemmaTokenizer.from_dir(GEMMA_DIR)

    # Load FeatureExtractorV2 + Connector
    fx = FeatureExtractorV2(embedding_dim=3840, out_features=2048, num_layers=49)
    load_feature_extractor_weights(fx, state)
    conn = Embeddings1DConnector(
        num_attention_heads=32, attention_head_dim=64, num_layers=8,
        num_learnable_registers=128, positional_embedding_max_pos=4096,
        seq_len=1024,
    )
    load_connector_weights(conn, state)

    proc = EmbeddingsProcessor(feature_extractor=fx, audio_connector=conn)
    encoder = DramaBoxPromptEncoder(gemma=gemma, tokenizer=tokenizer, processor=proc)

    encoded = encoder.encode('A woman speaks clearly.', max_length=1024)
    assert encoded.a_ctx.shape == (1, 1024, 2048)
    assert encoded.attention_mask.shape == (1, 1024)
    # No NaN/Inf in the final a_ctx
    assert mx.all(mx.isfinite(encoded.a_ctx)).item()
