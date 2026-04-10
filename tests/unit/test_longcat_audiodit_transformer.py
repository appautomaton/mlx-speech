from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models.longcat_audiodit.config import LongCatAudioDiTConfig
from mlx_speech.models.longcat_audiodit.transformer import LongCatAudioDiTTransformer


def test_tiny_transformer_returns_latent_shaped_hidden_state() -> None:
    config = LongCatAudioDiTConfig(
        latent_dim=4,
        dit_dim=16,
        dit_depth=2,
        dit_heads=4,
        dit_ff_mult=2.0,
        dit_text_dim=8,
        dit_qk_norm=True,
        dit_cross_attn=True,
        dit_cross_attn_norm=False,
        dit_text_conv=True,
        dit_use_latent_condition=True,
    )
    model = LongCatAudioDiTTransformer(config)
    x = mx.zeros((1, 6, 4), dtype=mx.float32)
    text = mx.zeros((1, 5, 8), dtype=mx.float32)
    text_len = mx.array([5], dtype=mx.int32)
    time = mx.array([0.5], dtype=mx.float32)
    mask = mx.array([[1, 1, 1, 1, 1, 0]], dtype=mx.bool_)
    cond_mask = mx.array([[1, 1, 1, 1, 1]], dtype=mx.bool_)
    latent_cond = mx.zeros((1, 6, 4), dtype=mx.float32)

    output = model(
        x=x,
        text=text,
        text_len=text_len,
        time=time,
        mask=mask,
        cond_mask=cond_mask,
        latent_cond=latent_cond,
        return_ith_layer=1,
    )

    assert output["last_hidden_state"].shape == (1, 6, 4)
    assert output["hidden_state"].shape == (1, 6, 16)


def test_transformer_parameter_tree_matches_checkpoint_naming() -> None:
    config = LongCatAudioDiTConfig(
        latent_dim=4,
        dit_dim=16,
        dit_depth=1,
        dit_heads=4,
        dit_ff_mult=2.0,
        dit_text_dim=8,
        dit_qk_norm=True,
        dit_cross_attn=True,
        dit_cross_attn_norm=False,
        dit_text_conv=True,
        dit_use_latent_condition=True,
    )
    model = LongCatAudioDiTTransformer(config)
    params = tree_flatten(model.parameters(), destination={})

    assert "time_embed.time_mlp.0.weight" in params
    assert "input_embed.proj.0.weight" in params
    assert "text_embed.proj.0.weight" in params
    assert "adaln_global_mlp.mlp.1.weight" in params
    assert "blocks.0.adaln_scale_shift" in params
    assert "blocks.0.self_attn.to_q.weight" in params
    assert "blocks.0.self_attn.q_norm.weight" in params
    assert "blocks.0.cross_attn.to_q.weight" in params
    assert "blocks.0.ffn.ff.0.weight" in params
    assert "text_conv_layer.0.dwconv.weight" in params
    assert "text_conv_layer.0.grn.gamma" in params
    assert "latent_embed.proj.0.weight" in params
    assert "latent_cond_embedder.proj.0.weight" in params
    assert "norm_out.linear.weight" in params
    assert "proj_out.weight" in params
