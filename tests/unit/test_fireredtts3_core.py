from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fireredtts3.core import (
    FireRedTTS3Core,
    FireRedTTS3CoreConfig,
)
from mlx_speech.models.fireredtts3.patch_encoder import PatchEncoder
from mlx_speech.models.fireredtts3.tokenizer import FireRedTTS3Tokenizer


def _tiny_config() -> FireRedTTS3CoreConfig:
    return FireRedTTS3CoreConfig(
        redae_dim=4,
        num_history_patches=2,
        spk_in_dim=6,
        patch_size=2,
        patch_encoder_hidden_size=8,
        patch_encoder_mlp_ratio=2,
        patch_encoder_depth=1,
        patch_encoder_num_heads=2,
        dit_mlp_ratio=2,
        dit_depth=1,
        dit_num_heads=2,
        dit_hidden_size=8,
        qwen_hidden_size=8,
        qwen_intermediate_size=16,
        qwen_num_hidden_layers=1,
        qwen_num_attention_heads=2,
        qwen_num_key_value_heads=1,
        qwen_head_dim=4,
        qwen_vocab_size=32,
        qwen_max_position_embeddings=64,
        qwen_rope_theta=10_000.0,
    )


class _Encoding:
    ids = [7, 8, 9]


class _TokenizerBackend:
    def __init__(self) -> None:
        self.special_tokens: list[str] = []
        self.source: str | None = None

    def add_special_tokens(self, tokens: list[str]) -> None:
        self.special_tokens.extend(tokens)

    def encode(self, source: str, *, add_special_tokens: bool) -> _Encoding:
        assert not add_special_tokens
        self.source = source
        return _Encoding()


def test_tokenizer_builds_exact_base_sequence_and_token_order() -> None:
    backend = _TokenizerBackend()
    tokenizer = FireRedTTS3Tokenizer(backend)
    ids = tokenizer.encode(
        language="Chinese",
        reference_text="参考",
        text="你好",
    )
    assert ids == [7, 8, 9]
    assert backend.source == "<|Chinese|><|sot|>参考你好<|eot|>"
    assert backend.special_tokens[7:9] == ["<|sot|>", "<|eot|>"]
    assert backend.special_tokens[207] == "<|Chinese|>"
    assert backend.special_tokens[-3:] == [
        "<|edit|>",
        "<|frame_patch|>",
        "<|end_edit|>",
    ]


def test_tokenizer_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        FireRedTTS3Tokenizer.build_input(
            language="Klingon",
            reference_text="reference",
            text="target",
        )


def test_patch_encoder_aggregates_four_latents_per_token() -> None:
    mx.random.seed(3)
    encoder = PatchEncoder(
        in_dim=4,
        out_dim=12,
        patch_size=4,
        hidden_size=8,
        mlp_ratio=2,
        depth=1,
        num_heads=2,
    )
    encoder.eval()
    latents = mx.arange(32, dtype=mx.float32).reshape(1, 8, 4) / 32
    first = encoder(latents)
    second = encoder(latents)
    mx.eval(first, second)
    assert first.shape == (1, 2, 12)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_tiny_core_uses_cache_and_removes_only_dummy_history() -> None:
    mx.random.seed(11)
    core = FireRedTTS3Core(_tiny_config())
    core.eval()
    speaker = mx.linspace(-0.2, 0.2, 6, dtype=mx.float32)[None]
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    prompt = mx.linspace(-0.1, 0.1, 32, dtype=mx.float32).reshape(1, 8, 4)
    first = core.generate(
        speaker_embedding=speaker,
        text_tokens=tokens,
        prompt_latents=prompt,
        flow_steps=2,
        guidance_scale=1.0,
        stop_threshold=2.0,
        max_generated_patches=2,
        seed=17,
    )
    second = core.generate(
        speaker_embedding=speaker,
        text_tokens=tokens,
        prompt_latents=prompt,
        flow_steps=2,
        guidance_scale=1.0,
        stop_threshold=2.0,
        max_generated_patches=2,
        seed=17,
    )
    mx.eval(first.latents, second.latents)
    assert first.generated_patches == 2
    assert first.latents.shape == (1, 12, 4)
    assert first.cache_length == first.prompt_length + 1
    np.testing.assert_allclose(first.latents[:, :8], prompt, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.latents, second.latents, atol=0.0, rtol=0.0)


def test_compiled_dit_matches_uncompiled_fixed_shape() -> None:
    mx.random.seed(31)
    core = FireRedTTS3Core(_tiny_config())
    core.eval()
    value = mx.random.normal((2, 6, 18))
    timestep = mx.array([0.25, 0.25])
    expected = core.dit(value, timestep)
    compiled = mx.compile(core.dit)
    actual = compiled(value, timestep)
    mx.eval(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
