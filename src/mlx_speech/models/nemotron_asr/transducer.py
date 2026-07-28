"""RNN-T prediction, joint, and greedy decoding for Nemotron 3.5 ASR."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import JointArgs, PredictArgs

LSTMState = tuple[mx.array, mx.array]


class MultiLayerLSTM(nn.Module):
    """Stacked MLX LSTMs with checkpoint-compatible list paths."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = [
            nn.LSTM(input_size if index == 0 else hidden_size, hidden_size)
            for index in range(num_layers)
        ]

    def __call__(
        self, x: mx.array, state: LSTMState | None = None
    ) -> tuple[mx.array, LSTMState]:
        if x.ndim != 3:
            raise ValueError(f"expected LSTM input [B, L, D], got {x.shape}")
        if state is None:
            hidden = [None] * self.num_layers
            cell = [None] * self.num_layers
        else:
            hidden_values, cell_values = state
            hidden = [hidden_values[index] for index in range(self.num_layers)]
            cell = [cell_values[index] for index in range(self.num_layers)]

        output = x
        next_hidden = []
        next_cell = []
        for index, layer in enumerate(self.lstm):
            all_hidden, all_cell = layer(
                output, hidden=hidden[index], cell=cell[index]
            )
            output = all_hidden
            next_hidden.append(all_hidden[:, -1])
            next_cell.append(all_cell[:, -1])
        return output, (
            mx.stack(next_hidden, axis=0),
            mx.stack(next_cell, axis=0),
        )


class PredictionNetwork(nn.Module):
    """RNN-T label-history network under NeMo's ``decoder.prediction`` path."""

    def __init__(self, args: PredictArgs) -> None:
        super().__init__()
        self.pred_hidden = args.pred_hidden
        embedding_size = args.vocab_size + int(args.blank_as_pad)
        self.prediction = {
            "embed": nn.Embedding(embedding_size, args.pred_hidden),
            "dec_rnn": MultiLayerLSTM(
                args.pred_hidden, args.pred_hidden, args.pred_rnn_layers
            ),
        }

    def __call__(
        self,
        token: mx.array | None,
        state: LSTMState | None = None,
    ) -> tuple[mx.array, LSTMState]:
        if token is None:
            batch = 1 if state is None else state[0].shape[1]
            embedded = mx.zeros((batch, 1, self.pred_hidden))
        else:
            embedded = self.prediction["embed"](token)
        return self.prediction["dec_rnn"](embedded, state)


class JointNetwork(nn.Module):
    """Project one acoustic frame and one prediction state to token logits."""

    def __init__(self, args: JointArgs) -> None:
        super().__init__()
        self.num_classes = args.num_classes + 1
        self.enc = nn.Linear(args.encoder_hidden, args.joint_hidden)
        self.pred = nn.Linear(args.pred_hidden, args.joint_hidden)
        activation_name = args.activation.lower()
        if activation_name == "relu":
            activation: nn.Module = nn.ReLU()
        elif activation_name == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"unsupported joint activation: {args.activation}")
        # Identity retains NeMo's inference-time ``joint_net.2`` weight path.
        self.joint_net = [
            activation,
            nn.Identity(),
            nn.Linear(args.joint_hidden, self.num_classes),
        ]

    def __call__(self, encoder: mx.array, prediction: mx.array) -> mx.array:
        if encoder.shape[:-1] != prediction.shape[:-1]:
            raise ValueError("joint inputs must have the same leading dimensions")
        output = self.enc(encoder) + self.pred(prediction)
        for layer in self.joint_net:
            output = layer(output)
        return output


@dataclass(frozen=True)
class GreedyDecodeResult:
    tokens: tuple[int, ...]
    frame_indices: tuple[int, ...]


def greedy_transducer_decode(
    encoded: mx.array,
    decoder: PredictionNetwork,
    joint: JointNetwork,
    *,
    blank_id: int,
    max_symbols: int = 10,
) -> GreedyDecodeResult:
    """Decode one utterance without materializing a ``T × U`` lattice."""
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise ValueError("greedy RNN-T decoding requires encoded shape [1, T, D]")
    if max_symbols < 1:
        raise ValueError("max_symbols must be positive")

    prediction, proposed_state = decoder(None, None)
    prediction = prediction[:, -1:, :].astype(encoded.dtype)
    tokens: list[int] = []
    frames: list[int] = []
    time = 0
    symbols_at_frame = 0
    while time < encoded.shape[1]:
        logits = joint(encoded[:, time : time + 1], prediction)
        token = int(mx.argmax(logits[0, 0]).item())
        if token == blank_id:
            time += 1
            symbols_at_frame = 0
            continue

        tokens.append(token)
        frames.append(time)
        symbols_at_frame += 1
        prediction, proposed_state = decoder(
            mx.array([[token]], dtype=mx.int32), proposed_state
        )
        prediction = prediction[:, -1:, :].astype(encoded.dtype)
        if symbols_at_frame >= max_symbols:
            time += 1
            symbols_at_frame = 0

    return GreedyDecodeResult(tuple(tokens), tuple(frames))


__all__ = [
    "GreedyDecodeResult",
    "JointNetwork",
    "LSTMState",
    "MultiLayerLSTM",
    "PredictionNetwork",
    "greedy_transducer_decode",
]
