"""Drama 3: the shared prompt head is prefilled once and reused, and dense int8 picks its layers."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_speech.generation.fish_drama_3 import FishDrama3Runtime, _head_length
from mlx_speech.models.fish_drama_3 import FishDrama3Config
from mlx_speech.models.fish_drama_3.model import FishDrama3Model, is_dense_int8_target
from tests.unit.test_fish_drama_3_config import tiny_payload

TURN = 7  # <|im_start|> in the stub vocabulary


class StubTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return TURN if token == "<|im_start|>" else None


def tiny_model() -> FishDrama3Model:
    payload = tiny_payload(fp8=False)
    payload["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    mx.random.seed(0)
    model = FishDrama3Model(FishDrama3Config.from_dict(payload))
    model.set_dtype(mx.bfloat16)  # as the converted checkpoint
    mx.eval(model.parameters())
    return model


def text(*ids: int) -> np.ndarray:
    rows = np.zeros((len(ids), 4), dtype=np.int32)
    rows[:, 0] = ids
    return rows


def audio(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 64, size=(frames, 3))
    return np.concatenate([codes[:, :1] + 200, codes], axis=1).astype(np.int32)


# System turn with a reference, a history turn, then the current turn and the opening.
HEAD = np.concatenate([text(TURN, 11, 12), audio(40, 1), text(150, TURN, 13), text(150, TURN, 14), audio(20, 2), text(150)])


def prompt(*words: int) -> np.ndarray:
    return np.concatenate([HEAD, text(TURN, 15, *words, 150), text(TURN, 16, 17)])


def generate(runtime: FishDrama3Runtime, rows: np.ndarray, seed: int) -> np.ndarray:
    mx.random.seed(seed)
    codes, _ = runtime.generate_frames(rows, max_new_tokens=6, temperature=0.7, top_k=30, top_p=0.7)
    return codes


def test_the_head_ends_where_the_current_user_turn_starts():
    assert _head_length(prompt(20, 21), TURN) == len(HEAD)
    assert _head_length(np.concatenate([text(TURN, 15, 150), text(TURN, 16)]), TURN) == 0


def test_a_reused_head_gives_the_same_frames_as_a_fresh_one():
    model = tiny_model()
    warm = FishDrama3Runtime(model=model, tokenizer=StubTokenizer(), codec=None)
    generate(warm, prompt(20, 21), seed=0)
    head = warm._head
    reused = generate(warm, prompt(22, 23, 24), seed=1)
    assert warm._head is head  # the second prompt did not prefill the head again
    assert reused.shape[1] > 0

    cold = FishDrama3Runtime(model=model, tokenizer=StubTokenizer(), codec=None)
    assert np.array_equal(generate(cold, prompt(22, 23, 24), seed=1), reused)


def test_a_different_head_is_prefilled_anew():
    runtime = FishDrama3Runtime(model=tiny_model(), tokenizer=StubTokenizer(), codec=None)
    generate(runtime, prompt(20), seed=0)
    first = runtime._head
    other = prompt(20).copy()
    other[5, 1] = (other[5, 1] + 1) % 64  # one reference code differs
    generate(runtime, other, seed=0)
    assert runtime._head is not first


def test_dense_int8_packs_the_projections_and_leaves_gates_heads_and_embeddings():
    model = tiny_model()
    nn.quantize(model, group_size=64, bits=8, class_predicate=is_dense_int8_target)
    packed = {path for path, module in model.named_modules() if type(module) is nn.QuantizedLinear}
    plain = {path for path, module in model.named_modules() if type(module) is nn.Linear}

    assert "backbone.layers.0.linear_attn.in_proj_qkv" in packed
    assert "backbone.layers.1.self_attn.o_proj" in packed
    assert "backbone.layers.0.mlp.shared_expert.down_proj" in packed
    assert "audio_decoder.residual_layers.0.transformer_layer.feed_forward.w2" in packed
    assert "audio_decoder.shared_layers.1.eh_proj" in packed
    for path in (
        "backbone.layers.0.linear_attn.in_proj_a",
        "backbone.layers.0.linear_attn.in_proj_b",
        "backbone.layers.0.mlp.gate",
        "backbone.layers.0.mlp.shared_expert_gate",
        "backbone.lm_head",
        "audio_decoder.codebook_heads.0",
    ):
        assert path in plain
