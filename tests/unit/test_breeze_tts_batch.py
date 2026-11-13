"""Independent EOS, random streams and padded attention in Breeze batches."""

from types import SimpleNamespace
from dataclasses import replace

import mlx.core as mx
import pytest

from mlx_speech.generation.breeze_tts_batch import (
    _decode_audio,
    _frame_keys,
    _generate_codes,
    _prefill,
    _sample_rows,
    generate_breeze_batch,
)
from mlx_speech.models.breeze_tts.backbone import BreezeBackbone
from mlx_speech.models.breeze_tts.config import BreezeDecoderConfig
from mlx_speech.models.breeze_tts.depth import BreezeDepthDecoder


def _config():
    decoder = BreezeDecoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=1000.0,
        max_position_embeddings=32,
        qk_norm=False,
        cache_growth=4,
    )
    return SimpleNamespace(
        backbone=decoder,
        depth=decoder,
        audio_embed_size=8,
        audio_embedding_rows=3 * 2051,
        num_codebooks=3,
        vocab_size=2051,
        backbone_eos_id=2051,
    )


def test_padded_prefill_and_cached_decode_match_individual_rows():
    # Compare float32 math without Metal's different batched attention rounding.
    with mx.stream(mx.cpu):
        _check_padded_decode()


def test_shared_prefix_prefill_preserves_padding_and_next_decode():
    with mx.stream(mx.cpu):
        config = _config()
        backbone = BreezeBackbone(config)
        speech = SimpleNamespace(config=config, backbone_model=backbone)
        prefix = mx.random.normal((3, 8))
        # Include a padding gap longer than the prefix itself.
        prompts = [
            mx.concatenate([prefix, mx.random.normal((length, 8))])
            for length in (1, 3, 7)
        ]
        expected, original, positions, pads = _prefill(speech, prompts, 4)
        actual, reused, reused_positions, reused_pads = _prefill(
            speech, prompts, 4, prefix_length=len(prefix)
        )
        assert mx.allclose(actual, expected, atol=1e-5).item()
        assert mx.array_equal(positions, reused_positions).item()
        assert mx.array_equal(pads, reused_pads).item()
        next_input = mx.random.normal((3, 1, 8))
        expected = backbone(
            next_input, original, position_ids=positions[:, None], pad_lengths=pads
        )
        actual = backbone(
            next_input, reused, position_ids=positions[:, None], pad_lengths=pads
        )
        assert mx.allclose(actual, expected, atol=1e-5).item()


@pytest.mark.parametrize("prefix_length", [0, 210])
def test_metal_bfloat16_padding_queries_do_not_poison_valid_hidden_states(prefix_length):
    config = _config()
    config.backbone = replace(
        config.backbone,
        hidden_size=128,
        intermediate_size=256,
        head_dim=128,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=512,
        qk_norm=True,
    )
    config.audio_embed_size = 128
    backbone = BreezeBackbone(config)
    backbone.set_dtype(mx.bfloat16)
    speech = SimpleNamespace(config=config, backbone_model=backbone)
    prefix = mx.random.normal((210, 128)).astype(mx.bfloat16)
    prompts = [
        mx.concatenate([prefix, mx.random.normal((length, 128)).astype(mx.bfloat16)])
        for length in (14, 15, 10)
    ]
    hidden, cache, _, _ = _prefill(speech, prompts, 4, prefix_length=prefix_length)
    assert mx.all(mx.isfinite(hidden)).item()
    assert all(mx.all(mx.isfinite(layer.values)).item() for layer in cache)


def _check_padded_decode():
    config = _config()
    backbone = BreezeBackbone(config)
    speech = SimpleNamespace(config=config, backbone_model=backbone)
    prompts = [mx.random.normal((2, 8)), mx.random.normal((5, 8))]
    hidden, cache, positions, pads = _prefill(speech, prompts, 4)
    next_input = mx.random.normal((2, 1, 8))
    next_hidden = backbone(
        next_input, cache, position_ids=positions[:, None], pad_lengths=pads
    )
    for i, prompt in enumerate(prompts):
        individual = backbone.make_cache(batch_size=1, dtype=prompt.dtype)
        expected = backbone(prompt[None], individual)
        assert mx.allclose(hidden[i], expected[0, -1], atol=1e-5).item()
        expected_next = backbone(next_input[i : i + 1], individual)
        assert mx.allclose(next_hidden[i], expected_next[0], atol=1e-5).item()


def test_request_random_keys_survive_batch_reordering_and_keep_top_k_candidates():
    keys = mx.stack([mx.random.key(i) for i in range(64)])
    logits = mx.broadcast_to(mx.array([0.0, 1.0, 1.0, -2.0]), (64, 4))
    draws = _sample_rows(logits, keys, 1.0, 2)
    kept = mx.array([31, 2, 49, 0])
    reordered = _sample_rows(logits[kept], keys[kept], 1.0, 2)
    assert mx.array_equal(draws[kept], reordered).item()
    assert set(draws.tolist()) == {1, 2}
    split = _frame_keys(keys, 4)
    assert mx.array_equal(split[kept], _frame_keys(keys[kept], 4)).item()


@pytest.mark.parametrize("guided", [False, True])
def test_eos_removes_only_its_request_and_does_not_touch_global_random_state(guided):
    config = _config()

    class Head:
        calls = 0

        def __call__(self, hidden):
            count = len(hidden) // (2 if guided else 1)
            tokens = [2051, 4] if self.calls == 0 else [2051]
            self.calls += 1
            logits = mx.full((count, 2052), -1000.0)
            logits[mx.arange(count), mx.array(tokens)] = 1000.0
            return mx.concatenate([logits, logits]) if guided else logits

    speech = SimpleNamespace(
        config=config,
        backbone_model=BreezeBackbone(config),
        depth_decoder=BreezeDepthDecoder(config),
        lm_head=Head(),
    )
    prompts = [mx.ones((2, 8)), mx.ones((4, 8))]
    if guided:
        prompts += [mx.ones((1, 8)), mx.ones((3, 8))]
    mx.random.seed(123)
    expected_random = mx.random.normal((8,))
    mx.eval(expected_random)
    mx.random.seed(123)
    codes = _generate_codes(
        speech,
        prompts,
        [1, 2],
        budgets=[3, 3],
        temperature=0.9,
        top_k=1,
        repetition_penalty=1.1,
        cfg_scale=4.0 if guided else 1.0,
    )
    assert [row.shape for row in codes] == [(0, 3), (1, 3)]
    assert codes[1][0, 0].item() == 4
    assert mx.array_equal(mx.random.normal((8,)), expected_random).item()


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("budget_finish", [False, True])
def test_quarter_compaction_preserves_codes_and_finished_lengths(
    monkeypatch, guided, budget_finish
):
    from mlx_speech.generation import breeze_tts_batch as batch

    finish_steps = (0, 4, 1, 6, 2, 5, 3, 7)
    coarse_residents = (
        list(range(8)), list(range(8)), [1, 3, 4, 5, 6, 7],
        [1, 3, 4, 5, 6, 7], [1, 3, 5, 7], [3, 5, 7], [3, 7], [7], [],
    )
    budgets = [9] * 8
    if budget_finish:
        budgets[4] = 2  # This row must remain finished while awaiting compaction.
    branches = 2 if guided else 1

    class Head:
        def residents(self, step):
            if self.coarse:
                return coarse_residents[step]
            return [i for i, finish in enumerate(finish_steps) if finish >= step]

        def __call__(self, hidden):
            self.step = self.calls
            self.calls += 1
            ids = self.residents(self.step)
            assert len(hidden) == len(ids) * branches
            # EOS is a one-step pulse: finished rows must never revive later.
            tokens = [
                2051 if finish_steps[i] == self.step and not (budget_finish and i == 4)
                else i + 1
                for i in ids
            ]
            logits = mx.full((len(ids), 2052), -1000.0)
            logits[mx.arange(len(ids)), mx.array(tokens)] = 1000.0
            return mx.concatenate([logits, logits]) if guided else logits

    head = Head()

    class Backbone(BreezeBackbone):
        def __call__(self, inputs, *args, **kwargs):
            self.batch_sizes.append(len(inputs) // branches)
            return super().__call__(inputs, *args, **kwargs)

    class Depth(BreezeDepthDecoder):
        def embed_code(self, ids, codebook):
            if codebook == 0:
                residents = head.residents(head.step + 1)
                expected = [
                    i + 1 if finish_steps[i] > head.step else 0 for i in residents
                ]
                assert ids.tolist() == expected
            return super().embed_code(ids, codebook)

    gathers = []
    select_cache = batch._select_cache

    def checked_select(cache, rows):
        before = head.residents(head.step)
        keep = [j for j, i in enumerate(before) if finish_steps[i] > head.step]
        assert keep  # Stopping the last row must not gather an empty cache.
        expected = keep + [j + len(before) for j in keep] if guided else keep
        assert rows.tolist() == expected
        gathers.append(len(rows) // branches)
        select_cache(cache, rows)

    monkeypatch.setattr(batch, "_select_cache", checked_select)
    with mx.stream(mx.cpu):
        mx.random.seed(11)
        config = _config()
        backbone = Backbone(config)
        speech = SimpleNamespace(
            config=config, backbone_model=backbone, depth_decoder=Depth(config),
            lm_head=head,
        )
        prompts = [
            mx.random.normal((2 + i % 3, 8), key=mx.random.key(100 + i))
            for i in range(8 * branches)
        ]

        def run(fraction):
            monkeypatch.setattr(batch, "_COMPACT_FINISHED_FRACTION", fraction)
            head.coarse, head.calls = fraction != 0, 0
            backbone.batch_sizes = []
            gathers.clear()
            codes = _generate_codes(
                speech, prompts, [13 + 7 * i for i in range(8)], budgets=budgets,
                temperature=0.9, top_k=50, repetition_penalty=1.1,
                cfg_scale=4.0 if guided else 1.0,
            )
            assert head.calls == 8
            return codes, list(backbone.batch_sizes), list(gathers)

        reference, _, _ = run(0.0)  # The former per-EOS compaction schedule.
        actual, sizes, compacted = run(0.25)
        assert [len(row) for row in actual] == list(finish_steps)
        for i, (row, expected) in enumerate(zip(actual, reference, strict=True)):
            assert mx.array_equal(row, expected).item()
            assert mx.all(row[:, 0] == i + 1).item()
        assert sizes == [8, 8, 6, 6, 4, 3, 2, 1]
        assert len(set(sizes)) == 6
        assert compacted == [n for n in (6, 4, 3, 2, 1) for _ in range(2)]
        assert [i for i, row in enumerate(actual) if len(row) == budgets[i]] == (
            [4] if budget_finish else []
        )


@pytest.mark.parametrize(
    "controls, message",
    [
        ({"seeds": [1]}, "seeds"),
        ({"batch_size": 0}, "batch_size"),
        ({"cfg_scale": 4}, "instruction"),
        ({"top_k": -1}, "top_k"),
        ({"max_frames": 0}, "max_frames"),
    ],
)
def test_invalid_batch_controls_fail_before_loading_or_inference(controls, message):
    with pytest.raises(ValueError, match=message):
        generate_breeze_batch(None, None, None, ["one", "two"], **controls)


def test_empty_batch_is_empty_and_single_string_is_rejected():
    assert generate_breeze_batch(None, None, None, []) == []
    with pytest.raises(ValueError, match="sequence"):
        generate_breeze_batch(None, None, None, "not a batch")


def test_codec_batches_bound_padded_frames_for_mixed_output_lengths():
    class Codec:
        calls = []

        def decode(self, codes):
            self.calls.append((codes.shape[0], codes.shape[2]))
            return mx.zeros((codes.shape[0], 1, codes.shape[2] * 1920))

    codec = Codec()
    lengths = [1, 120, 3, 100, 2, 119]
    output = _decode_audio(codec, [mx.zeros((n, 16), mx.int32) for n in lengths])
    assert [item.frames for item in output] == lengths
    assert all(rows * frames <= 256 for rows, frames in codec.calls)
    assert [item.samples.size for item in output] == [n * 1920 for n in lengths]


def test_budget_warning_names_input_indices_across_partitions(monkeypatch):
    from mlx_speech.generation import breeze_tts_batch as batch

    def codes(speech, prompts, seeds, *, budgets, **_):
        frames = budgets[0] if seeds == [8] else 1  # Only the second input is capped.
        return [mx.zeros((frames, 16), mx.int32)]

    monkeypatch.setattr(batch, "_prepare_prompts", lambda *args: [None])
    monkeypatch.setattr(batch, "_generate_codes", codes)
    monkeypatch.setattr(batch, "_decode_audio", lambda codec, rows: rows)
    with pytest.warns(RuntimeWarning, match=r"requests \[1\] reached their frame"):
        generate_breeze_batch(
            None, None, None, ["one", "two"], seeds=[7, 8], max_frames=3, batch_size=1
        )


def test_each_request_stops_at_its_own_frame_budget():
    config = _config()

    class Head:
        def __call__(self, hidden):  # Never EOS: every request runs to its budget.
            logits = mx.full((len(hidden), 2052), -1000.0)
            logits[:, 4] = 1000.0
            return logits

    speech = SimpleNamespace(
        config=config,
        backbone_model=BreezeBackbone(config),
        depth_decoder=BreezeDepthDecoder(config),
        lm_head=Head(),
    )
    codes = _generate_codes(
        speech,
        [mx.ones((2, 8)), mx.ones((4, 8)), mx.ones((3, 8))],
        [1, 2, 3],
        budgets=[3, 1, 2],
        temperature=0.9,
        top_k=1,
        repetition_penalty=1.1,
        cfg_scale=1.0,
    )
    assert [row.shape for row in codes] == [(3, 3), (1, 3), (2, 3)]


def test_partitions_follow_budgets_and_results_keep_input_order(monkeypatch):
    from mlx_speech.generation import breeze_tts_batch as batch

    calls = []

    def codes(speech, prompts, seeds, *, budgets, **_):
        calls.append((seeds, budgets))
        return [mx.zeros((b - 1, 16), mx.int32) for b in budgets]

    monkeypatch.setattr(batch, "_prepare_prompts", lambda *args: [None])
    monkeypatch.setattr(batch, "_generate_codes", codes)
    monkeypatch.setattr(batch, "_decode_audio", lambda codec, rows: rows)
    output = generate_breeze_batch(
        None, None, None, ["a", "b", "c"], max_frames=[5, 2, 3], batch_size=2
    )
    assert calls == [([1, 2], [2, 3]), ([0], [5])]
    assert [len(row) for row in output] == [4, 1, 2]


def test_frame_budgets_must_match_the_texts():
    with pytest.raises(ValueError, match="one per text"):
        generate_breeze_batch(None, None, None, ["a", "b"], max_frames=[3])
