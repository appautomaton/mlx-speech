from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fish_drama_3.model import load_fish_drama_3_model

MODEL_DIR = Path("models/fish_drama_3/mlx-int8")


def test_converted_checkpoint_fills_the_model_exactly():
    if not (MODEL_DIR / "config.json").is_file():
        pytest.skip("No local converted Drama 3 checkpoint")

    # Lazy: strict loading checks every name and shape without reading 42 GB.
    model = load_fish_drama_3_model(MODEL_DIR, materialize=False)

    experts = model.backbone.layers[0].mlp.experts.gate_proj
    assert experts.weight.dtype == mx.uint32
    assert experts.scales.dtype == mx.bfloat16
    assert model.backbone.layers[3].self_attn.q_proj.weight.dtype == mx.bfloat16
    assert model.backbone.layers[0].linear_attn.conv1d.weight.shape == (8192, 4, 1)
    assert len(model.audio_decoder.residual_layers) == 9
    assert (model.audio_decoder.main_candidates[1:] - 248099).tolist() == list(
        range(4096)
    )
    assert (MODEL_DIR / "codec-mlx" / "model.safetensors").is_file()


def test_real_tokenizer_agrees_with_the_config():
    if not (MODEL_DIR / "tokenizer.json").is_file():
        pytest.skip("No local converted Drama 3 checkpoint")
    from tokenizers import Tokenizer

    from mlx_speech.models.fish_drama_3 import FishDrama3Config
    from mlx_speech.models.fish_drama_3.prompt import build_prompt

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rows = build_prompt(
        "<|speaker:15|> Hi. <|speaker:1|> Hello.",
        tokenizer,
        FishDrama3Config.from_path(MODEL_DIR),
    )

    speaker_15 = tokenizer.token_to_id("<|speaker:15|>")
    assert speaker_15 is not None and speaker_15 in rows[:, 0].tolist()
