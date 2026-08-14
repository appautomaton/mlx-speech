import json
from pathlib import Path

from mlx_speech.models.moss_local import MossTTSLocalConfig


def test_moss_local_config_from_dict_exposes_derived_properties() -> None:
    config = MossTTSLocalConfig.from_dict(
        {
            "n_vq": 32,
            "audio_vocab_size": 1024,
            "local_hidden_size": 1536,
            "local_num_layers": 4,
            "language_config": {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 151943,
                "max_position_embeddings": 4096,
            },
        }
    )

    assert config.hidden_size == 4096
    assert config.vocab_size == 151943
    assert config.channels == 33
    assert config.audio_embedding_vocab_size == 1025
    assert config.language_config.effective_head_dim == 128


def test_moss_local_config_from_path_reads_config_json(tmp_path: Path) -> None:
    payload = {
        "sampling_rate": 24000,
        "audio_start_token_id": 151652,
        "language_config": {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151943,
            "max_position_embeddings": 4096,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = MossTTSLocalConfig.from_path(tmp_path)

    assert config.sampling_rate == 24000
    assert config.audio_start_token_id == 151652
    assert config.language_config.num_hidden_layers == 24
