"""Pure Step-Audio flow math and tiny random-weight shape tests."""

from __future__ import annotations

import numpy as np

from mlx_speech.models.step_audio_editx import (
    StepAudioCausalConditionalCFM,
    StepAudioCausalMaskedDiffWithXvec,
    StepAudioFlowConditioner,
    StepAudioFlowConditioningConfig,
    StepAudioUpsampleConformerEncoderV2,
    interpolate_prompt_features,
    reshape_mixed_audio_tokens,
)
from mlx_speech.models.step_audio_editx.flow_model import StepAudioDiT


def test_reshape_mixed_audio_tokens_matches_shipped_grouping() -> None:
    reshaped = reshape_mixed_audio_tokens([10, 11, 1024, 1025, 1026])

    expected = np.asarray(
        [
            [10, 1025],
            [11, 1026],
            [1024, 1027],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(reshaped, expected)


def test_reshape_mixed_audio_tokens_matches_shipped_padding_behavior() -> None:
    reshaped = reshape_mixed_audio_tokens([10, 11, 1024, 1025, 1026, 20, 21])

    expected = np.asarray(
        [
            [10, 1025],
            [11, 1026],
            [1024, 1027],
            [20, 1025],
            [21, 1025],
            [1024, 1025],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(reshaped, expected)


def test_interpolate_prompt_features_uses_nearest_neighbor_time_repeat() -> None:
    prompt_feat = np.asarray([[[1.0], [3.0]]], dtype=np.float32)

    interpolated = interpolate_prompt_features(prompt_feat, target_length=6)

    expected = np.asarray([[[1.0], [1.0], [1.0], [3.0], [3.0], [3.0]]], dtype=np.float32)
    assert np.array_equal(interpolated, expected)


def test_flow_conditioner_prepare_nonstream_inputs_shapes() -> None:
    config = StepAudioFlowConditioningConfig(
        vocab_size=64,
        input_size=8,
        output_size=6,
        spk_embed_dim=4,
    )
    model = StepAudioFlowConditioner(config)

    prepared = model.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029],
        prompt_token=[0, 1, 1024, 1025, 1026],
        prompt_feat=np.ones((1, 4, 6), dtype=np.float32),
        speaker_embedding=np.ones((1, 4), dtype=np.float32),
    )

    assert prepared.prompt_token_dual.shape == (1, 3, 2)
    assert prepared.token_dual.shape == (1, 3, 2)
    assert prepared.concatenated_token_dual.shape == (1, 6, 2)
    assert prepared.concatenated_token_length.tolist() == [6]
    assert prepared.embedded_tokens.shape == (1, 6, 8)
    assert prepared.prompt_feat_aligned.shape == (1, 6, 6)
    assert prepared.projected_speaker_embedding.shape == (1, 6)


def test_tiny_flow_model_inference_returns_expected_shape() -> None:
    conditioner = StepAudioFlowConditioner(
        StepAudioFlowConditioningConfig(
            vocab_size=64,
            input_size=8,
            output_size=6,
            spk_embed_dim=4,
        )
    )
    prepared = conditioner.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029],
        prompt_token=[0, 1, 1024, 1025, 1026],
        prompt_feat=np.ones((1, 4, 6), dtype=np.float32),
        speaker_embedding=np.ones((1, 4), dtype=np.float32),
    )

    encoder = StepAudioUpsampleConformerEncoderV2(
        input_size=8,
        output_size=8,
        pre_lookahead_len=2,
        num_blocks=1,
        num_up_blocks=1,
        up_stride=2,
        up_scale_factor=2.0,
        attention_heads=2,
        linear_units=16,
        key_bias=True,
    )
    estimator = StepAudioDiT(
        in_channels=24,
        out_channels=6,
        mlp_ratio=2.0,
        depth=1,
        num_heads=2,
        head_dim=4,
        hidden_size=8,
    )
    decoder = StepAudioCausalConditionalCFM(estimator, inference_cfg_rate=0.7)
    model = StepAudioCausalMaskedDiffWithXvec(
        input_size=8,
        output_size=6,
        spk_embed_dim=4,
        vocab_size=64,
        encoder=encoder,
        decoder=decoder,
        input_embedding=conditioner.input_embedding,
    )

    mel = model.inference(prepared, n_timesteps=2)

    assert mel.shape == (1, 6, 6)
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()
