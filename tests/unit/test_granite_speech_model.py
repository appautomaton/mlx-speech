from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

import mlx_speech.models.granite_speech_asr.model as model_module
from mlx_speech.models.granite_speech_asr.checkpoint import (
    AlignmentReport,
    GraniteSpeechCheckpoint,
    load_checkpoint_into_model,
)
from mlx_speech.models.granite_speech_asr.config import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
    GraniteSpeechTextConfig,
)
from mlx_speech.models.granite_speech_asr.model import (
    GraniteSpeechModel,
    GraniteSpeechModelBundle,
)


def _tiny_config() -> GraniteSpeechConfig:
    return GraniteSpeechConfig(
        encoder=GraniteSpeechEncoderConfig(
            input_dim=8,
            hidden_dim=16,
            output_dim=5,
            num_layers=2,
            num_heads=2,
            dim_head=4,
            feedforward_mult=2,
            conv_expansion_factor=2,
            conv_kernel_size=3,
            context_size=4,
            max_pos_emb=8,
            dropout=0.0,
        ),
        projector=GraniteSpeechProjectorConfig(
            hidden_size=16,
            encoder_hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            layer_norm_eps=1e-6,
        ),
        text=GraniteSpeechTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        ),
        audio_token_index=31,
        window_size=4,
        downsample_rate=2,
    )


def _write_tokenizer_assets(model_dir: Path, *, audio_token_id: int = 31) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<|unk|>": 0,
                "USER:": 1,
                "ASSISTANT:": 2,
                "hello": 3,
                "<|audio|>": audio_token_id,
            },
            unk_token="<|unk|>",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(["<|audio|>"])
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "added_tokens.json").write_text(
        json.dumps({"<|audio|>": audio_token_id}),
        encoding="utf-8",
    )
    (model_dir / "chat_template.jinja").write_text(
        "USER: {{ message['content'] }}\n ASSISTANT:",
        encoding="utf-8",
    )


def test_granite_model_composes_audio_and_text_paths():
    model = GraniteSpeechModel(_tiny_config())
    features = mx.zeros((1, 5, 8), dtype=mx.float32)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)

    audio_features = model.get_audio_features(features)
    logits = model(input_ids)
    mx.eval(audio_features, logits)

    assert audio_features.shape == (1, 4, 16)
    assert logits.shape == (1, 3, 32)


def test_granite_model_parameter_names_match_checkpoint_roots():
    params = tree_flatten(GraniteSpeechModel(_tiny_config()).parameters(), destination={})
    expected = {
        "encoder.input_linear.weight",
        "encoder.layers.0.conv.batch_norm.running_var",
        "projector.query",
        "projector.qformer.encoder.layer.0.crossattention.attention.key.weight",
        "projector.linear.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.1.mlp.down_proj.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    }

    assert expected <= set(params)


def test_granite_load_checkpoint_rejects_partial_strict_alignment(tmp_path):
    model = GraniteSpeechModel(_tiny_config())
    checkpoint = GraniteSpeechCheckpoint(
        model_dir=tmp_path,
        config=_tiny_config(),
        state_dict={"checkpoint.only": mx.zeros((1,), dtype=mx.float32)},
        source_files=(tmp_path / "model.safetensors",),
        skipped_keys=(),
        transposed_keys=(),
    )

    with pytest.raises(ValueError, match="checkpoint-only"):
        load_checkpoint_into_model(model, checkpoint, strict=True)


def test_granite_model_from_dir_loads_components_without_retaining_state_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _tiny_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    _write_tokenizer_assets(tmp_path, audio_token_id=config.audio_token_index)

    checkpoint = GraniteSpeechCheckpoint(
        model_dir=tmp_path,
        config=config,
        state_dict={"unused": mx.zeros((1,), dtype=mx.float32)},
        source_files=(tmp_path / "model.safetensors",),
        skipped_keys=("encoder.layers.0.conv.batch_norm.num_batches_tracked",),
        transposed_keys=("encoder.layers.0.conv.up_conv.weight",),
    )
    calls: dict[str, object] = {}

    def fake_load_checkpoint(model_dir):
        calls["model_dir"] = Path(model_dir)
        return checkpoint

    def fake_load_into_model(model: nn.Module, ckpt: GraniteSpeechCheckpoint, *, strict: bool):
        calls["model"] = model
        calls["checkpoint"] = ckpt
        calls["strict"] = strict
        return AlignmentReport(checkpoint_only=(), model_only=(), shape_mismatches=())

    monkeypatch.setattr(model_module, "load_granite_speech_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(model_module, "load_checkpoint_into_model", fake_load_into_model)

    loaded = GraniteSpeechModel.from_dir(tmp_path, dtype=mx.float32, strict=False)

    assert isinstance(loaded, GraniteSpeechModelBundle)
    assert isinstance(loaded.model, GraniteSpeechModel)
    assert loaded.config == config
    assert loaded.source_files == checkpoint.source_files
    assert loaded.skipped_keys == checkpoint.skipped_keys
    assert loaded.transposed_keys == checkpoint.transposed_keys
    assert calls["model_dir"] == tmp_path
    assert calls["strict"] is False
    assert not hasattr(loaded, "checkpoint")
    assert not hasattr(loaded, "state_dict")
