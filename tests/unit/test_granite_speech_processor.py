from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.granite_speech_asr.processor import (
    mask_audio_token_ids,
    replace_audio_embeddings,
)


def test_replace_audio_embeddings_substitutes_audio_positions_only():
    input_ids = mx.array([[10, 100352, 11, 100352]], dtype=mx.int32)
    token_embeddings = mx.array(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=mx.float32,
    )
    audio_features = mx.array([[[20.0, 21.0], [30.0, 31.0]]], dtype=mx.float32)

    merged = replace_audio_embeddings(
        input_ids,
        token_embeddings,
        audio_features,
        audio_token_id=100352,
    )

    np.testing.assert_allclose(
        np.array(merged),
        np.array([[[1.0, 1.0], [20.0, 21.0], [3.0, 3.0], [30.0, 31.0]]], dtype=np.float32),
    )


def test_replace_audio_embeddings_rejects_mismatched_audio_token_counts():
    with pytest.raises(ValueError, match="audio token count mismatch"):
        replace_audio_embeddings(
            mx.array([[100352, 100352]], dtype=mx.int32),
            mx.zeros((1, 2, 4), dtype=mx.float32),
            mx.zeros((1, 1, 4), dtype=mx.float32),
            audio_token_id=100352,
        )


def test_mask_audio_token_ids_replaces_sentinel_before_text_embedding():
    ids = mx.array([[5, 100352, 7]], dtype=mx.int32)

    masked = mask_audio_token_ids(ids, audio_token_id=100352, replacement_id=0)

    np.testing.assert_array_equal(np.array(masked), np.array([[5, 0, 7]], dtype=np.int32))
