"""Breeze rejects invalid sampling controls and does not truncate silently."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_speech.generation import breeze_tts
from mlx_speech.generation.breeze_tts import _iter_code_frames

EOS = 2051


class _Backbone:
    def make_cache(self, *, batch_size: int, dtype: mx.Dtype) -> list:
        del batch_size, dtype
        return []

    def __call__(self, inputs: mx.array, cache: list) -> mx.array:
        del cache
        return mx.zeros((1, int(inputs.shape[1]), 4))

    def embed_audio(self, codes: mx.array) -> mx.array:
        return mx.zeros((1, int(codes.shape[1]), 4))


class _Speech:
    """Backbone logits always favour one token: a code, or EOS."""

    def __init__(self, favoured: int) -> None:
        self.config = SimpleNamespace(backbone_eos_id=EOS)
        self.backbone_model = _Backbone()
        self._logits = mx.zeros((1, EOS + 1)).at[0, favoured].add(1.0e6)

    def lm_head(self, hidden: mx.array) -> mx.array:
        del hidden
        return self._logits


@pytest.fixture
def stub_prompt_and_depth(monkeypatch) -> None:
    monkeypatch.setattr(
        breeze_tts, "_prompt_embeddings", lambda *args, **kwargs: mx.zeros((1, 3, 4))
    )
    monkeypatch.setattr(
        breeze_tts, "_depth_frame", lambda *args, **kwargs: mx.zeros((16,), mx.int32)
    )


@pytest.mark.parametrize(
    ("control", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -0.5}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_k": 5.0}, "top_k"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"repetition_penalty": float("inf")}, "repetition_penalty"),
    ],
)
def test_invalid_sampling_controls_are_rejected(control: dict, message: str) -> None:
    frames = _iter_code_frames(object(), object(), object(), "hello", **control)
    with pytest.raises(ValueError, match=message):
        next(frames)


@pytest.mark.usefixtures("stub_prompt_and_depth")
def test_frame_budget_without_eos_warns() -> None:
    with pytest.warns(RuntimeWarning, match="3-frame budget"):
        frames = list(
            _iter_code_frames(_Speech(7), object(), object(), "hello", max_frames=3)
        )
    assert len(frames) == 3


@pytest.mark.usefixtures("stub_prompt_and_depth")
def test_eos_before_the_budget_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frames = list(
            _iter_code_frames(_Speech(EOS), object(), object(), "hello", max_frames=3)
        )
    assert frames == []
