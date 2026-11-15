from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.fish_drama_3.model import load_fish_drama_3_model

# The upstream release, loaded as is: its FP8 experts are requantized in memory.
MODEL_DIR = Path("models/fish_drama_3/base")
# Relative error of the int8 experts against the release's FP8 values, measured over
# all 120 projections (0.57% mean, 0.66% worst on the 2026-10-29 release);
# mx.quantize's own range, applied to the bf16 rounding, left 0.76% on average.
MAX_MEAN_EXPERT_ERROR = 0.006
MAX_EXPERT_ERROR = 0.007


def _skip_without_release():
    if not (MODEL_DIR / "config.json").is_file():
        pytest.skip("No local Drama 3 release")


def test_release_fills_the_model_and_keeps_the_fp8_experts_close():
    _skip_without_release()
    from mlx_speech.checkpoints.sharded import ShardedCheckpointIndex
    from mlx_speech.models.fish_drama_3.checkpoint import (
        EXPERT_PROJECTIONS,
        dequantize_fp8_blocks,
    )

    # Strict: every name and shape checked; the experts are requantized now, the
    # dense weights are read on first use.
    model = load_fish_drama_3_model(MODEL_DIR, materialize=False)
    assert model.backbone.layers[0].mlp.experts.gate_proj.weight.dtype == mx.uint32
    assert model.backbone.layers[0].mlp.experts.gate_proj.scales.dtype == mx.bfloat16
    assert model.backbone.layers[3].self_attn.q_proj.weight.dtype == mx.bfloat16
    assert model.backbone.layers[0].linear_attn.conv1d.weight.shape == (8192, 4, 1)
    assert len(model.audio_decoder.residual_layers) == 9
    assert (model.audio_decoder.main_candidates[1:] - 248099).tolist() == list(
        range(4096)
    )

    index = ShardedCheckpointIndex.from_directory(MODEL_DIR)

    def fp8_stack(base: str, proj: str) -> mx.array:
        """One projection's exact FP8 values, read beside the loaded model."""
        shards: dict[str, dict[str, mx.array]] = {}

        def tensor(key: str) -> mx.array:
            name = index.weight_map[key]
            if name not in shards:
                shards[name] = mx.load(str(MODEL_DIR / name))
            return shards[name].pop(key)

        return dequantize_fp8_blocks(
            mx.stack([tensor(f"{base}.{e}.{proj}.weight") for e in range(256)]),
            mx.stack([tensor(f"{base}.{e}.{proj}.weight_scale_inv") for e in range(256)]),
        )

    errors = []
    for layer_index, layer in enumerate(model.backbone.layers):
        base = f"model.language_model.layers.{layer_index}.mlp.experts"
        for proj in EXPERT_PROJECTIONS:
            exact = fp8_stack(base, proj)
            linear = getattr(layer.mlp.experts, proj)
            held = mx.dequantize(
                linear.weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
            ).astype(mx.float32)
            error = mx.sqrt(((held - exact) ** 2).sum() / (exact**2).sum())
            errors.append(error.item())
            del exact, held
            mx.clear_cache()

    assert len(errors) == 120
    assert sum(errors) / len(errors) < MAX_MEAN_EXPERT_ERROR
    assert max(errors) < MAX_EXPERT_ERROR


def test_codec_builds_from_the_release_codec_pth_in_memory():
    _skip_without_release()
    from mlx_speech.models.fish_s2_pro.codec import FishS2Codec

    codec = FishS2Codec.from_codec_pth(MODEL_DIR / "codec.pth")
    assert codec.sample_rate == 44100

    seconds = 1.0
    t = mx.arange(int(seconds * codec.sample_rate)) / codec.sample_rate
    waveform = 0.3 * mx.sin(2 * 3.141592653589793 * 220.0 * t)
    codes = codec.encode(waveform)
    assert codes.shape[0] == 10  # the semantic codebook and nine residual ones
    audio = codec.decode(codes).reshape(-1)
    assert audio.shape[0] == codes.shape[1] * 2048
    assert mx.isfinite(audio).all().item()


def test_real_tokenizer_agrees_with_the_config():
    _skip_without_release()
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
