import mlx.core as mx

from mlx_voice.generation import (
    MossTTSLocalGenerationConfig,
    extract_audio_code_sequences,
    generate_moss_tts_local,
    sample_next_token,
)
from mlx_voice.generation.moss_local import _can_use_kv_cache, _resolve_sampling_config
from mlx_voice.generation.moss_local import _resolve_generation_limit
from mlx_voice.models.moss_local import (
    MossTTSLocalConfig,
    MossTTSLocalModel,
    estimate_duration_tokens,
)


def _tiny_config() -> MossTTSLocalConfig:
    return MossTTSLocalConfig.from_dict(
        {
            "n_vq": 2,
            "audio_vocab_size": 16,
            "audio_pad_code": 16,
            "local_hidden_size": 12,
            "local_ffn_hidden_size": 32,
            "additional_mlp_ffn_hidden_size": 24,
            "local_num_layers": 2,
            "language_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 128,
                "max_position_embeddings": 64,
            },
        }
    )


def test_extract_audio_code_sequences_skips_all_pad_rows() -> None:
    sequences = mx.array(
        [
            [
                [10, 1024, 1024],
                [20, 1, 2],
                [21, 3, 4],
                [151653, 1024, 1024],
            ]
        ],
        dtype=mx.int32,
    )

    extracted = extract_audio_code_sequences(
        sequences,
        prompt_length=1,
        pad_code=1024,
        n_vq=2,
    )

    assert len(extracted) == 1
    assert extracted[0].shape == (2, 2)
    assert extracted[0].tolist() == [[1, 2], [3, 4]]


def test_extract_audio_code_sequences_stops_before_audio_end_row() -> None:
    sequences = mx.array(
        [
            [
                [10, 1024, 1024],
                [151652, 1024, 1024],
                [151656, 1, 2],
                [151656, 3, 4],
                [151653, 9, 9],
            ]
        ],
        dtype=mx.int32,
    )

    extracted = extract_audio_code_sequences(
        sequences,
        prompt_length=1,
        pad_code=1024,
        n_vq=2,
        stop_token=151653,
    )

    assert len(extracted) == 1
    assert extracted[0].tolist() == [[1, 2], [3, 4]]


def test_generate_moss_tts_local_appends_rows() -> None:
    model = MossTTSLocalModel(_tiny_config())
    input_ids = mx.array(
        [
            [
                [1, 16, 16],
                [2, 16, 16],
                [151652, 16, 16],
            ]
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    output = generate_moss_tts_local(
        model,
        input_ids,
        attention_mask,
        config=MossTTSLocalGenerationConfig(
            max_new_tokens=2,
            n_vq_for_inference=2,
            do_sample=False,
        ),
    )

    assert output.sequences.shape == (1, 5, 3)
    assert output.generated_rows.shape == (1, 2, 3)
    assert len(output.audio_codes_list) == 1
    assert output.audio_codes_list[0].ndim == 2
    assert output.audio_codes_list[0].shape[1] == 2


def test_sampling_config_uses_upstream_style_per_channel_defaults() -> None:
    config = MossTTSLocalGenerationConfig()

    text_cfg = _resolve_sampling_config(0, config)
    audio_cfg = _resolve_sampling_config(1, config)

    assert text_cfg == (1.5, 50, 1.0, 1.0, True)
    assert audio_cfg == (1.0, 50, 0.95, 1.1, True)


def test_app_default_generation_config_matches_upstream_gradio_defaults() -> None:
    config = MossTTSLocalGenerationConfig.app_defaults()

    assert config.audio_temperature == 1.7
    assert config.audio_top_p == 0.8
    assert config.audio_top_k == 25
    assert config.audio_repetition_penalty == 1.0


def test_estimate_duration_tokens_matches_upstream_heuristic() -> None:
    assert estimate_duration_tokens("Hello") == ("en", 4, 2, 6)
    assert estimate_duration_tokens("你好世界") == ("zh", 12, 6, 18)


def test_sample_next_token_greedy_ignores_sampling_warpers() -> None:
    logits = mx.array([[0.1, 0.2, 0.9, 0.8]], dtype=mx.float32)
    previous = mx.array([[2, 3, 1]], dtype=mx.int32)

    token = sample_next_token(
        logits,
        previous_tokens=previous,
        temperature=1.7,
        top_k=1,
        top_p=0.2,
        repetition_penalty=2.0,
        do_sample=False,
    )

    assert token.tolist() == [2]


def test_kv_cache_only_enables_for_single_item_batches() -> None:
    config = MossTTSLocalGenerationConfig(use_kv_cache=True)
    single = mx.zeros((1, 3, 3), dtype=mx.int32)
    batch = mx.zeros((2, 3, 3), dtype=mx.int32)

    assert MossTTSLocalGenerationConfig().use_kv_cache is True
    assert _can_use_kv_cache(single, config) is True
    assert _can_use_kv_cache(batch, config) is False
    assert (
        _can_use_kv_cache(
            single,
            MossTTSLocalGenerationConfig(use_kv_cache=False),
        )
        is False
    )
    assert (
        _can_use_kv_cache(
            single,
            MossTTSLocalGenerationConfig(use_kv_cache=True, do_sample=False),
        )
        is False
    )


def test_resolve_generation_limit_uses_safety_cap_when_user_limit_is_unset() -> None:
    assert (
        _resolve_generation_limit(
            MossTTSLocalGenerationConfig(
                max_new_tokens=None,
                safety_max_new_tokens=77,
            )
        )
        == 77
    )
