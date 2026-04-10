import json
import subprocess
import sys

import pytest

from mlx_speech.models.fish_s2_pro.config import (
    FishAudioDecoderConfig,
    FishS2ProConfig,
    FishTextConfig,
)


UPSTREAM_CONFIG_PAYLOAD = {
    "model_type": "fish_qwen3_omni",
    "dtype": "bfloat16",
    "sample_rate": 44100,
    "pad_token_id": 151669,
    "eos_token_id": 151645,
    "audio_pad_token_id": 151677,
    "semantic_start_token_id": 151678,
    "semantic_end_token_id": 155773,
    "text_config": {
        "vocab_size": 155776,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 10,
        "head_dim": 160,
        "hidden_size": 3200,
        "intermediate_size": 12800,
        "rope_base": 500_000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 16384,
        "attention_qkv_bias": False,
        "attention_o_bias": False,
        "attention_qk_norm": True,
    },
    "audio_decoder_config": {
        "vocab_size": 4096,
        "num_hidden_layers": 6,
        "num_attention_heads": 20,
        "num_key_value_heads": 5,
        "head_dim": 160,
        "hidden_size": 3200,
        "intermediate_size": 12800,
        "rope_base": 500_000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 13,
        "attention_qkv_bias": False,
        "attention_o_bias": False,
        "attention_qk_norm": False,
        "text_dim": 3200,
        "num_codebooks": 12,
    },
}


def test_config_defaults_match_real_fish_s2_shape():
    cfg = FishS2ProConfig()
    assert cfg.sample_rate == 44100
    assert cfg.text_config.vocab_size == 155776
    assert cfg.text_config.n_layer == 36
    assert cfg.text_config.dim == 2560
    assert cfg.audio_decoder_config.vocab_size == 4096
    assert cfg.audio_decoder_config.n_layer == 4
    assert cfg.audio_decoder_config.num_codebooks == 10


def test_config_from_dict_reads_nested_upstream_fields():
    cfg = FishS2ProConfig.from_dict(UPSTREAM_CONFIG_PAYLOAD)
    assert cfg.eos_token_id == 151645
    assert cfg.semantic_start_token_id == 151678
    assert cfg.semantic_end_token_id == 155773
    assert cfg.text_config.max_seq_len == 16384
    assert cfg.text_config.n_layer == 40
    assert cfg.text_config.n_head == 40
    assert cfg.text_config.n_local_heads == 10
    assert cfg.text_config.dim == 3200
    assert cfg.audio_decoder_config.max_seq_len == 13
    assert cfg.audio_decoder_config.n_layer == 6
    assert cfg.audio_decoder_config.n_head == 20
    assert cfg.audio_decoder_config.n_local_heads == 5
    assert cfg.audio_decoder_config.dim == 3200
    assert cfg.audio_decoder_config.num_codebooks == 12


def test_config_from_path_reads_nested_upstream_fields(tmp_path):
    model_dir = tmp_path / "fish_s2_pro"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(UPSTREAM_CONFIG_PAYLOAD), encoding="utf-8"
    )

    cfg = FishS2ProConfig.from_path(model_dir)
    assert cfg.model_dir == str(model_dir)
    assert cfg.text_config.max_seq_len == 16384
    assert cfg.audio_decoder_config.max_seq_len == 13


def test_config_from_dict_preserves_explicit_falsey_numeric_values():
    payload = {
        "text_config": {
            "vocab_size": 0,
            "num_hidden_layers": 0,
            "num_attention_heads": 0,
            "num_key_value_heads": 0,
            "head_dim": 0,
            "hidden_size": 0,
            "intermediate_size": 0,
            "rope_base": 0.0,
            "rms_norm_eps": 0.0,
            "max_position_embeddings": 0,
        },
        "audio_decoder_config": {
            "vocab_size": 0,
            "num_hidden_layers": 0,
            "num_attention_heads": 0,
            "num_key_value_heads": 0,
            "head_dim": 0,
            "hidden_size": 0,
            "intermediate_size": 0,
            "rope_base": 0.0,
            "rms_norm_eps": 0.0,
            "max_position_embeddings": 0,
            "text_dim": 0,
            "num_codebooks": 0,
        },
    }

    cfg = FishS2ProConfig.from_dict(payload)

    assert cfg.text_config.vocab_size == 0
    assert cfg.text_config.n_layer == 0
    assert cfg.text_config.n_head == 0
    assert cfg.text_config.n_local_heads == 0
    assert cfg.text_config.head_dim == 0
    assert cfg.text_config.dim == 0
    assert cfg.text_config.intermediate_size == 0
    assert cfg.text_config.rope_base == 0.0
    assert cfg.text_config.norm_eps == 0.0
    assert cfg.text_config.max_seq_len == 0
    assert cfg.audio_decoder_config.vocab_size == 0
    assert cfg.audio_decoder_config.n_layer == 0
    assert cfg.audio_decoder_config.n_head == 0
    assert cfg.audio_decoder_config.n_local_heads == 0
    assert cfg.audio_decoder_config.head_dim == 0
    assert cfg.audio_decoder_config.dim == 0
    assert cfg.audio_decoder_config.intermediate_size == 0
    assert cfg.audio_decoder_config.rope_base == 0.0
    assert cfg.audio_decoder_config.norm_eps == 0.0
    assert cfg.audio_decoder_config.max_seq_len == 0
    assert cfg.audio_decoder_config.text_dim == 0
    assert cfg.audio_decoder_config.num_codebooks == 0


def test_package_import_exposes_config_classes_without_runtime_imports():
    command = (
        "import sys; "
        "from mlx_speech.models.fish_s2_pro import FishS2ProConfig, FishTextConfig, FishAudioDecoderConfig; "
        "assert FishS2ProConfig.__name__ == 'FishS2ProConfig'; "
        "assert FishTextConfig.__name__ == 'FishTextConfig'; "
        "assert FishAudioDecoderConfig.__name__ == 'FishAudioDecoderConfig'; "
        "assert 'mlx_speech.models.fish_s2_pro.llama' not in sys.modules; "
        "assert 'mlx_speech.models.fish_s2_pro.tokenizer' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_import_exposes_real_tokenizer_class_and_missing_file_error():
    command = "\n".join(
        [
            "import sys",
            "from mlx_speech.models.fish_s2_pro import FishS2Tokenizer",
            "assert FishS2Tokenizer.__name__ == 'FishS2Tokenizer'",
            "assert FishS2Tokenizer.__module__ == 'mlx_speech.models.fish_s2_pro.tokenizer'",
            "assert 'mlx_speech.models.fish_s2_pro.tokenizer' in sys.modules",
            "try:",
            "    FishS2Tokenizer.from_pretrained('__missing_fish_tokenizer_dir__')",
            "except FileNotFoundError as exc:",
            "    assert 'tokenizer.json' in str(exc)",
            "else:",
            "    raise AssertionError('expected missing tokenizer.json error')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_import_reraises_non_dependency_tokenizer_import_errors(monkeypatch):
    import mlx_speech.models.fish_s2_pro as fish_s2_pro

    def fake_import_module(name, package):
        if (name, package) == (".tokenizer", "mlx_speech.models.fish_s2_pro"):
            raise ImportError("tokenizer module bug")
        return __import__(name)

    monkeypatch.setattr(fish_s2_pro.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="tokenizer module bug"):
        fish_s2_pro.__getattr__("FishS2Tokenizer")


def test_package_import_reraises_module_not_found_errors_from_tokenizer_import(
    monkeypatch,
):
    import mlx_speech.models.fish_s2_pro as fish_s2_pro

    def fake_import_module(name, package):
        if (name, package) == (".tokenizer", "mlx_speech.models.fish_s2_pro"):
            raise ModuleNotFoundError(
                "No module named 'transformers'", name="transformers"
            )
        return __import__(name)

    monkeypatch.setattr(fish_s2_pro.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="transformers"):
        fish_s2_pro.__getattr__("FishS2Tokenizer")


def test_config_round_trips_to_dict():
    cfg = FishS2ProConfig(
        model_type="custom_fish",
        dtype="float32",
        sample_rate=16000,
        pad_token_id=0,
        eos_token_id=1,
        audio_pad_token_id=2,
        semantic_start_token_id=3,
        semantic_end_token_id=4,
        model_dir="models/fish_s2_pro/custom",
        text_config=FishTextConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_local_heads=1,
            head_dim=8,
            dim=32,
            intermediate_size=64,
            rope_base=0.0,
            norm_eps=0.0,
            max_seq_len=16,
            attention_qkv_bias=True,
            attention_o_bias=True,
            attention_qk_norm=False,
        ),
        audio_decoder_config=FishAudioDecoderConfig(
            vocab_size=200,
            n_layer=3,
            n_head=5,
            n_local_heads=2,
            head_dim=9,
            dim=33,
            intermediate_size=65,
            rope_base=0.0,
            norm_eps=0.0,
            max_seq_len=17,
            attention_qkv_bias=True,
            attention_o_bias=True,
            attention_qk_norm=True,
            text_dim=34,
            num_codebooks=6,
        ),
    )
    payload = cfg.to_dict()
    restored = FishS2ProConfig.from_dict(payload)
    assert restored == cfg
    assert restored.to_dict() == payload
