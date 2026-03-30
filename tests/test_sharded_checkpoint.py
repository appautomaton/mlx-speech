import json
from pathlib import Path

import mlx.core as mx

from mlx_voice.models.moss_local import load_moss_tts_local_checkpoint


def test_load_moss_local_checkpoint_from_sharded_safetensors(tmp_path: Path) -> None:
    config = {
        "n_vq": 32,
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
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    mx.save_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {
            "model.language_model.embed_tokens.weight": mx.ones((2, 2)),
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones((2, 3)),
        },
    )
    mx.save_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        {
            "local_transformer.layers.0.self_attn.q_proj.weight": mx.ones((3, 4)),
            "model.rotary_emb.inv_freq": mx.ones((8,)),
        },
    )
    index = {
        "metadata": {"total_size": 1234},
        "weight_map": {
            "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "local_transformer.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.rotary_emb.inv_freq": "model-00002-of-00002.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )

    checkpoint = load_moss_tts_local_checkpoint(tmp_path)

    assert len(checkpoint.source_files) == 2
    assert checkpoint.key_count == 3
    assert checkpoint.skipped_keys == ("model.rotary_emb.inv_freq",)
    assert checkpoint.renamed_keys == ()
    assert checkpoint.state_dict["model.language_model.embed_tokens.weight"].shape == (2, 2)
