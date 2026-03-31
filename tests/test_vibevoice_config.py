"""Tests for VibeVoice config parsing."""

from pathlib import Path

import pytest

from mlx_speech.models.vibevoice.config import (
    Qwen2LanguageConfig,
    VibeVoiceConfig,
    VibeVoiceConvTokenizerConfig,
)

MODEL_DIR = Path("models/vibevoice/original")
HAS_CHECKPOINT = (MODEL_DIR / "config.json").exists()


class TestQwen2LanguageConfig:
    def test_from_dict_round_trip(self):
        cfg = Qwen2LanguageConfig(
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            vocab_size=152064,
        )
        rt = Qwen2LanguageConfig.from_dict(cfg.to_dict())
        assert rt.hidden_size == 3584
        assert rt.head_dim == 128

    def test_extra_preserved(self):
        cfg = Qwen2LanguageConfig.from_dict({
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "torch_dtype": "bfloat16",
        })
        assert cfg.extra["torch_dtype"] == "bfloat16"


class TestConvTokenizerConfig:
    def test_parsed_depths(self):
        cfg = VibeVoiceConvTokenizerConfig(vae_dim=64)
        assert cfg.parsed_encoder_depths == [3, 3, 3, 3, 3, 3, 8]
        assert cfg.parsed_decoder_depths == [8, 3, 3, 3, 3, 3, 3]

    def test_ratios_as_tuple(self):
        cfg = VibeVoiceConvTokenizerConfig.from_dict({
            "vae_dim": 64,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
        })
        assert isinstance(cfg.encoder_ratios, tuple)


class TestVibeVoiceConfig:
    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint not available")
    def test_from_path(self):
        cfg = VibeVoiceConfig.from_path(MODEL_DIR)
        assert cfg.model_type == "vibevoice"
        assert cfg.hidden_size == 3584
        assert cfg.language_config.num_hidden_layers == 28
        assert cfg.acoustic_tokenizer_config.vae_dim == 64
        assert cfg.semantic_tokenizer_config.vae_dim == 128
        assert cfg.diffusion_config.head_layers == 4

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint not available")
    def test_round_trip(self):
        cfg = VibeVoiceConfig.from_path(MODEL_DIR)
        rt = VibeVoiceConfig.from_dict(cfg.to_dict())
        assert rt.hidden_size == cfg.hidden_size
        assert rt.acoustic_tokenizer_config.vae_dim == cfg.acoustic_tokenizer_config.vae_dim

    def test_acostic_typo_handled(self):
        """The upstream config.json has a typo: acostic_vae_dim."""
        cfg = VibeVoiceConfig.from_dict({
            "decoder_config": {
                "hidden_size": 256, "intermediate_size": 1024,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "vocab_size": 100,
            },
            "acoustic_tokenizer_config": {"vae_dim": 64},
            "semantic_tokenizer_config": {"vae_dim": 128},
            "diffusion_head_config": {},
            "acostic_vae_dim": 64,
        })
        assert cfg.acoustic_vae_dim == 64
