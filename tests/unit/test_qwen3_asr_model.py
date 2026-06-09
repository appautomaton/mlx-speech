from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRThinkerConfig,
)
from mlx_speech.models.qwen3_asr.model import (
    Qwen3ASRModel,
    count_audio_tokens,
    mask_audio_token_ids,
    replace_audio_embeddings,
)


def _config() -> Qwen3ASRConfig:
    audio = Qwen3ASRAudioConfig(
        d_model=16,
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        downsample_hidden_size=4,
        output_dim=16,
        max_source_positions=64,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    text = Qwen3ASRTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        eos_token_id=9,
        extra={"tie_word_embeddings": True},
    )
    return Qwen3ASRConfig(
        thinker_config=Qwen3ASRThinkerConfig(
            audio_config=audio,
            text_config=text,
            audio_token_id=31,
            audio_start_token_id=29,
            audio_end_token_id=30,
        ),
        support_languages=("Chinese", "English"),
    )


def test_qwen3_asr_model_encodes_audio_features_shape():
    model = Qwen3ASRModel(_config())
    input_features = mx.zeros((1, 8, 20), dtype=mx.float32)
    feature_attention_mask = mx.ones((1, 20), dtype=mx.int32)

    audio_features = model.get_audio_features(
        input_features,
        feature_attention_mask=feature_attention_mask,
    )
    mx.eval(audio_features)

    assert audio_features.shape == (3, 16)
    assert mx.all(mx.isfinite(audio_features)).item()


def test_qwen3_asr_mask_and_count_audio_token_ids():
    input_ids = mx.array([[1, 31, 31, 2]], dtype=mx.int32)

    masked = mask_audio_token_ids(input_ids, audio_token_id=31, replacement_id=0)
    mx.eval(masked)

    np.testing.assert_array_equal(np.array(masked), np.array([[1, 0, 0, 2]]))
    assert count_audio_tokens(input_ids, audio_token_id=31) == 2


def test_qwen3_asr_replace_audio_embeddings_in_prompt_positions():
    input_ids = mx.array([[1, 31, 31, 2]], dtype=mx.int32)
    token_embeddings = mx.zeros((1, 4, 16), dtype=mx.float32)
    audio_features = mx.ones((2, 16), dtype=mx.float32) * 99.0

    replaced = replace_audio_embeddings(
        input_ids,
        token_embeddings,
        audio_features,
        audio_token_id=31,
    )
    mx.eval(replaced)
    replaced_np = np.array(replaced)

    assert np.all(replaced_np[0, 1:3] == 99.0)
    assert np.all(replaced_np[0, [0, 3]] == 0.0)


def test_qwen3_asr_replace_audio_embeddings_rejects_placeholder_mismatch():
    input_ids = mx.array([[1, 31, 31, 2]], dtype=mx.int32)
    token_embeddings = mx.zeros((1, 4, 16), dtype=mx.float32)
    audio_features = mx.ones((1, 16), dtype=mx.float32)

    with pytest.raises(ValueError, match="placeholder count mismatch"):
        replace_audio_embeddings(
            input_ids,
            token_embeddings,
            audio_features,
            audio_token_id=31,
        )


def test_qwen3_asr_model_rejects_audio_text_width_mismatch():
    config = _config()
    bad_audio = Qwen3ASRAudioConfig(
        **{**config.audio_config.__dict__, "output_dim": 8, "extra": {}}
    )

    with pytest.raises(ValueError, match="output_dim"):
        Qwen3ASRModel(
            Qwen3ASRConfig(
                thinker_config=Qwen3ASRThinkerConfig(
                    audio_config=bad_audio,
                    text_config=config.text_config,
                    audio_token_id=31,
                    audio_start_token_id=29,
                    audio_end_token_id=30,
                )
            )
        )
