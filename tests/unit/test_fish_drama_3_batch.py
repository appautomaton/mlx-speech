"""Drama 3 batched decoding: each line decodes as it would alone, whatever its
neighbours, prompt length and frame budget."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech.generation.fish_drama_3 as generation
from mlx_speech.generation.fish_drama_3 import (
    Drama3Chunk,
    Drama3Reference,
    FishDrama3Runtime,
)
from mlx_speech.models.fish_drama_3 import FishDrama3Config
from mlx_speech.models.fish_drama_3.model import FishDrama3Model
from mlx_speech.models.fish_drama_3.prompt import (
    MAX_PROMPT_ROWS,
    PromptTooLongError,
    build_prompt,
)
from mlx_speech.tts._adapters.fish_drama_3 import FishDrama3Adapter
from tests.unit.test_fish_drama_3_config import tiny_payload
from tests.unit.test_fish_drama_3_prompt_head import StubTokenizer, prompt


@pytest.fixture(autouse=True)
def on_cpu():
    """fp32 on the CPU, where batched and single matmuls agree closely enough that
    no draw flips."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def runtime(codec=None) -> FishDrama3Runtime:
    payload = tiny_payload(fp8=False)
    payload["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    mx.random.seed(0)
    model = FishDrama3Model(FishDrama3Config.from_dict(payload))
    mx.eval(model.parameters())
    return FishDrama3Runtime(model=model, tokenizer=StubTokenizer(), codec=codec)


def decode(rt: FishDrama3Runtime, prompts, budgets, seeds):
    return rt._decode(
        [rt._prefill_prompt(rows) for rows in prompts],
        budgets,
        mx.stack([mx.random.key(seed) for seed in seeds]),
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )


# Prompts of different lengths, so their cached rows are padded differently.
PROMPTS = [prompt(20), prompt(20, 21, 22, 23, 24), prompt(20, 21)]


def test_each_batched_line_decodes_as_it_would_alone():
    rt = runtime()
    budgets, seeds = [9, 4, 12], [3, 4, 5]  # the 4-frame line leaves the batch early

    together = decode(rt, PROMPTS, budgets, seeds)

    for rows, budget, seed, (codes, reason) in zip(PROMPTS, budgets, seeds, together):
        alone, alone_reason = decode(rt, [rows], [budget], [seed])[0]
        assert codes.shape[1] > 0
        assert reason == alone_reason
        np.testing.assert_array_equal(codes, alone)


def test_a_line_stops_at_its_own_budget():
    rt = runtime()

    results = decode(rt, PROMPTS, [2, 5, 3], [0, 0, 0])

    for (codes, reason), budget in zip(results, [2, 5, 3]):
        assert codes.shape[1] <= budget
        assert reason == "length" or codes.shape[1] < budget


class Codec:
    def decode(self, codes):
        return mx.zeros((1, codes.shape[1] * 4))


def test_synthesize_batch_keeps_input_order_and_matches_lines_alone(monkeypatch):
    rt = runtime(Codec())
    by_text = {"a": PROMPTS[0], "b": PROMPTS[1], "c": PROMPTS[2]}
    monkeypatch.setattr(
        generation, "build_prompt", lambda text, *args, **kwargs: by_text[text]
    )

    lines = rt.synthesize_batch(
        ["c", "a", "b"], seeds=[7, 8, 9], max_new_tokens=[5, 8, 3], batch_size=2
    )

    assert [line.text for line in lines] == ["c", "a", "b"]
    for line, seed, budget in zip(lines, [7, 8, 9], [5, 8, 3]):
        alone = rt.synthesize_batch([line.text], seeds=[seed], max_new_tokens=budget)[0]
        np.testing.assert_array_equal(line.codes, alone.codes)
        assert line.waveform.shape == (line.codes.shape[1] * 4,)


def test_synthesize_batch_checks_every_line_before_generating(monkeypatch):
    rt = runtime(Codec())

    def build(text, *args, **kwargs):
        if text == "bad":
            raise ValueError("text must not be empty")
        return PROMPTS[0]

    monkeypatch.setattr(generation, "build_prompt", build)
    monkeypatch.setattr(rt, "_decode", lambda *a, **k: pytest.fail("generated"))

    with pytest.raises(ValueError, match="line 1"):
        rt.synthesize_batch(["fine", "bad"], seeds=[0, 1])
    with pytest.raises(ValueError, match="one entry per line"):
        rt.synthesize_batch(["fine", "fine"], seeds=[0, 1], duration_seconds=[1.0])


class FakeRuntime:
    sample_rate = 44_100

    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def synthesize_batch(self, texts, references, **kwargs):
        self.calls.append((texts, references, kwargs))
        return [
            Drama3Chunk(text, np.zeros((10, n), np.int32), mx.ones((n,)), reason)
            for text, (n, reason) in zip(texts, self.frames)
        ]


def test_adapter_seeds_lines_in_order_and_warns_about_cut_lines():
    rt = FakeRuntime([(3, "eos"), (5, "length"), (0, "eos")])

    with pytest.warns(RuntimeWarning) as caught:
        outputs = FishDrama3Adapter(rt).generate_batch(
            ["one", "two", "three"], seed=10, duration_seconds=[1.0, None, 2.0]
        )

    _, _, kwargs = rt.calls[0]
    assert kwargs["seeds"] == [10, 11, 12]
    assert kwargs["duration_seconds"] == [1.0, None, 2.0]
    assert [o.waveform.shape[0] for o in outputs] == [3, 5, 0]
    messages = " ".join(str(w.message) for w in caught)
    assert "[1] reached their frame budget" in messages
    assert "[2] stopped before generating any audio" in messages


def test_adapter_refuses_a_single_string():
    with pytest.raises(TypeError, match="sequence of strings"):
        FishDrama3Adapter(FakeRuntime([])).generate_batch("one line")


@pytest.mark.parametrize(
    ("count", "batch_size", "sizes"),
    [(130, 128, [65, 65]), (128, 128, [128]), (5, 2, [1, 2, 2]), (1, 128, [1])],
)
def test_lines_split_into_the_fewest_even_groups(monkeypatch, count, batch_size, sizes):
    rt = runtime(Codec())
    monkeypatch.setattr(generation, "build_prompt", lambda *a, **k: PROMPTS[0])
    seen = []
    monkeypatch.setattr(rt, "_prefill_prompt", lambda rows: None)

    def decode(requests, budgets, keys, **kwargs):
        seen.append(len(requests))
        return [(np.zeros((3, 0), np.int32), "eos")] * len(requests)

    monkeypatch.setattr(rt, "_decode", decode)
    rt.synthesize_batch(["x"] * count, seeds=list(range(count)), batch_size=batch_size)

    assert sorted(seen) == sorted(sizes)


class PromptTokenizer:
    """Deterministic stub tokenizer whose ids fit the tiny 300-token vocabulary."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def token_to_id(self, token: str) -> int | None:
        return {
            "<|im_start|>": 7,
            "<|im_end|>": 150,
            "<|semantic:0|>": 200,
            "<|semantic:63|>": 263,
        }.get(token)

    def encode(self, text: str) -> SimpleNamespace:
        if text.startswith("<|im_start|>"):
            return SimpleNamespace(ids=[7, self._id(text[len("<|im_start|>") :])])
        if text.startswith("<|im_end|>"):
            return SimpleNamespace(ids=[150, 8])
        token_id = self._id(text)
        return SimpleNamespace(ids=[token_id, token_id + 1])

    def _id(self, text: str) -> int:
        if text not in self._ids:
            self._ids[text] = 10 + (len(self._ids) * 2) % 128
        return self._ids[text]


def prompt_runtime(codec=None) -> FishDrama3Runtime:
    rt = runtime(codec)
    rt.tokenizer = PromptTokenizer()
    return rt


def test_synthesize_batch_forwards_history_reuses_head_and_matches_alone():
    rt = prompt_runtime(Codec())
    ref = Drama3Reference(codes=np.array([[1, 2], [3, 4], [5, 6]], np.int32), text="ref")
    history = [("prior turn", np.array([[7, 8, 9], [1, 2, 3], [4, 5, 6]], np.int32))]
    prefill_spans: list[tuple[int, int]] = []
    original_prefill = rt._prefill

    def track_prefill(rows, start, end, backbone_cache, decoder_cache):
        prefill_spans.append((start, end))
        return original_prefill(rows, start, end, backbone_cache, decoder_cache)

    rt._prefill = track_prefill  # type: ignore[method-assign]

    lines = rt.synthesize_batch(
        ["first line", "second line"],
        [ref],
        seeds=[3, 4],
        history=history,
        max_new_tokens=5,
        batch_size=2,
    )

    # Line 0 prefills head (start=0) + tail; line 1 reuses the cached head (start > 0).
    assert len(prefill_spans) == 3
    assert prefill_spans[0][0] == 0
    assert prefill_spans[1][0] == prefill_spans[0][1] > 0
    assert prefill_spans[2][0] == prefill_spans[0][1]

    cold = prompt_runtime(Codec())
    for text, seed, line in zip(["first line", "second line"], [3, 4], lines):
        rows = build_prompt(
            text, cold.tokenizer, cold.config, [ref.text], [ref.codes], history=history
        )
        alone, alone_reason = cold.generate_frames(
            rows,
            max_new_tokens=5,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            key=mx.random.key(seed),
        )
        assert line.finish_reason == alone_reason
        np.testing.assert_array_equal(line.codes, alone)


def test_synthesize_batch_preserves_prompt_too_long_error(monkeypatch):
    rt = prompt_runtime(Codec())
    monkeypatch.setattr(rt, "_decode", lambda *a, **k: pytest.fail("generated"))
    oversized = np.ones((3, MAX_PROMPT_ROWS), dtype=np.int32)

    with pytest.raises(PromptTooLongError, match="line 0"):
        rt.synthesize_batch(["hello"], seeds=[0], history=[("prior", oversized)])
