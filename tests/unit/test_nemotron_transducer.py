from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.nemotron_asr.config import JointArgs, PredictArgs, PromptArgs
from mlx_speech.models.nemotron_asr.prompt import (
    apply_language_prompt,
    build_prompt_kernel,
    resolve_prompt_index,
)
from mlx_speech.models.nemotron_asr.tokenizer import NemotronTokenizer
from mlx_speech.models.nemotron_asr.transducer import (
    JointNetwork,
    PredictionNetwork,
    greedy_transducer_decode,
)


def test_prediction_network_paths_and_state_shapes() -> None:
    network = PredictionNetwork(
        PredictArgs(pred_hidden=4, pred_rnn_layers=2, vocab_size=5)
    )
    names = {name for name, _ in tree_flatten(network.parameters())}

    output, (hidden, cell) = network(mx.array([[2]], dtype=mx.int32))
    mx.eval(output, hidden, cell)

    assert output.shape == (1, 1, 4)
    assert hidden.shape == (2, 1, 4)
    assert cell.shape == (2, 1, 4)
    assert "prediction.embed.weight" in names
    assert "prediction.dec_rnn.lstm.0.Wx" in names
    assert "prediction.dec_rnn.lstm.1.Wh" in names
    assert "prediction.dec_rnn.lstm.1.bias" in names


def test_joint_computes_one_lattice_cell() -> None:
    joint = JointNetwork(
        JointArgs(
            encoder_hidden=4,
            pred_hidden=4,
            joint_hidden=3,
            num_classes=5,
        )
    )

    logits = joint(mx.ones((1, 1, 4)), mx.zeros((1, 1, 4)))
    mx.eval(logits)

    assert logits.shape == (1, 1, 6)
    with pytest.raises(ValueError, match="leading dimensions"):
        joint(mx.ones((1, 2, 4)), mx.zeros((1, 1, 4)))


def test_greedy_blank_advances_time_and_token_does_not() -> None:
    class Decoder:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, token, state=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            value = mx.zeros((1, 1, 2))
            next_state = (mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)))
            return value, next_state

    class Joint:
        def __init__(self) -> None:
            self.tokens = iter((1, 3, 2, 3))

        def __call__(self, encoder, prediction):  # type: ignore[no-untyped-def]
            token = next(self.tokens)
            return mx.eye(4, dtype=mx.float32)[token][None, None]

    decoder = Decoder()
    result = greedy_transducer_decode(
        mx.zeros((1, 2, 2)),
        decoder,  # type: ignore[arg-type]
        Joint(),  # type: ignore[arg-type]
        blank_id=3,
        max_symbols=10,
    )

    assert result.tokens == (1, 2)
    assert result.frame_indices == (0, 1)
    assert decoder.calls == 3


def test_greedy_max_symbols_guard_forces_time_forward() -> None:
    class Decoder:
        def __call__(self, token, state=None):  # type: ignore[no-untyped-def]
            value = mx.zeros((1, 1, 2))
            next_state = (mx.zeros((1, 1, 2)), mx.zeros((1, 1, 2)))
            return value, next_state

    class AlwaysToken:
        def __call__(self, encoder, prediction):  # type: ignore[no-untyped-def]
            return mx.array([[[0.0, 1.0, 0.0]]])

    result = greedy_transducer_decode(
        mx.zeros((1, 2, 2)),
        Decoder(),  # type: ignore[arg-type]
        AlwaysToken(),  # type: ignore[arg-type]
        blank_id=2,
        max_symbols=2,
    )

    assert result.tokens == (1, 1, 1, 1)
    assert result.frame_indices == (0, 0, 1, 1)


def test_language_prompt_shape_and_alias_resolution() -> None:
    args = PromptArgs(
        num_prompts=3,
        prompt_hidden=8,
        prompt_dictionary={"en-US": 0, "auto": 2},
    )
    kernel = build_prompt_kernel(4, args)

    output = apply_language_prompt(mx.zeros((1, 5, 4)), "auto", args, kernel)
    mx.eval(output)

    assert output.shape == (1, 5, 4)
    assert resolve_prompt_index("en-US", args) == 0
    assert resolve_prompt_index("auto", args) == 2
    with pytest.raises(ValueError, match="unsupported"):
        resolve_prompt_index("xx-XX", args)


def test_tokenizer_decodes_pieces_and_detects_language() -> None:
    tokenizer = NemotronTokenizer(
        ("<unk>", "▁Hello", "▁world", ".", "<en-US>")
    )
    tokens = (1, 2, 3, 4)

    assert tokenizer.decode(tokens) == "Hello world."
    assert tokenizer.decode(tokens, strip_language_tags=False) == "Hello world.<en-US>"
    assert tokenizer.detected_language(tokens) == "en-US"
    assert tokenizer.is_special(4)
    assert not tokenizer.is_special(2)


def test_prediction_none_is_deterministic_for_same_state() -> None:
    network = PredictionNetwork(
        PredictArgs(pred_hidden=4, pred_rnn_layers=1, vocab_size=5)
    )

    first, first_state = network(None)
    second, second_state = network(None)
    mx.eval(first, second, first_state, second_state)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
