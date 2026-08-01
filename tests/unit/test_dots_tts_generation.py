from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.dots_tts import (
    DEFAULT_MAX_AUDIO_PATCHES,
    _RESAMPLE_WORKSPACE_BYTES,
    _DotsTTSRequestState,
    DotsTTSGenerator,
    DotsTTSPromptConditioning,
    DotsTTSSynthesisOutput,
    _build_fm_attention_mask,
    _high_quality_resample,
    _resample_plan,
)
from mlx_speech.models.dots_tts.text import DotsTTSTokenizer


class _Backend:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        self.encoded.append(text)
        return SimpleNamespace(ids=[10 + (ord(char) % 20) for char in text])


class _Projection:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.weight = mx.ones((output_size, input_size), dtype=dtype)

    def __call__(self, value: mx.array) -> mx.array:
        return value @ self.weight.T


class _Qwen:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.stop = False
        self.steps = 0

    def get_input_embeddings(self):
        return lambda ids: mx.repeat(
            ids[..., None].astype(mx.float32), self.hidden_size, -1
        )

    def step(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        cache=None,
        cache_capacity=None,
    ):
        del cache_capacity
        value = inputs_embeds
        if value is None:
            value = self.get_input_embeddings()(input_ids)
        self.steps += 1
        return SimpleNamespace(
            last_hidden_state=value + 0.01,
            cache=[] if cache is None else cache,
        )

    def should_stop(self, hidden, *, threshold: float):
        del hidden, threshold
        return mx.array([self.stop])


class _Semantic:
    def prefill(self, value, *, max_audio_patches=None):
        del max_audio_patches
        patches = int(value.shape[1]) // 2
        embedded = mx.repeat(
            mx.mean(value.reshape(1, patches, -1), -1)[..., None], 4, -1
        )
        return embedded, SimpleNamespace(count=patches)

    def decode_patch(self, value, state):
        embedded, _ = self.prefill(value)
        return embedded, SimpleNamespace(count=state.count + 1)


class _DiT:
    hidden_size = 4
    _dots_tts_full_history_test_double = True

    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        sequence,
        timesteps,
        *,
        duration=None,
        attention_mask=None,
        positions=None,
        speaker_condition=None,
    ):
        del timesteps, duration, attention_mask, positions, speaker_condition
        self.calls += 1
        return mx.full((*sequence.shape[:2], 2), 0.05, dtype=sequence.dtype)


class _LatentIO:
    def normalize(self, value):
        return value

    def denormalize(self, value):
        return value + 0.25


class _AudioVAE:
    def __init__(self):
        self.decode_calls: list[tuple[int, bool]] = []
        self.batch_latents: list[mx.array] = []
        self.stream_latents: list[mx.array] = []

    @staticmethod
    def _decode(latent):
        waveform = mx.repeat(mx.sum(latent, axis=1), 2, axis=1)
        return waveform[:, None]

    def decode(self, latent):
        self.batch_latents.append(latent)
        return self._decode(latent)

    def init_decode_state(self, *, maximum_chunk_size):
        return SimpleNamespace(maximum_chunk_size=maximum_chunk_size)

    def decode_chunk(self, latent, state, *, final=False):
        assert int(latent.shape[-1]) <= state.maximum_chunk_size
        self.decode_calls.append((int(latent.shape[-1]), final))
        self.stream_latents.append(latent)
        return self._decode(latent), state


def _generator(
    mode: str,
    activation_dtype: mx.Dtype = mx.float32,
) -> tuple[DotsTTSGenerator, _Backend, _DiT]:
    backend = _Backend()
    tokenizer = DotsTTSTokenizer(
        backend=backend,
        audio_gen_start_id=101,
        audio_gen_span_id=102,
        audio_gen_end_id=103,
        audio_comp_span_id=104,
        text_cond_end_id=105,
    )
    dit = _DiT()
    qwen = _Qwen(4)
    core = SimpleNamespace(
        qwen=qwen,
        semantic_encoder=_Semantic(),
        dit=dit,
        coordinate_projection=_Projection(2, 4, activation_dtype),
        hidden_projection=_Projection(4, 4, activation_dtype),
        latent_projection=_Projection(2, 4, activation_dtype),
    )
    config = SimpleNamespace(
        mode=mode,
        patch_size=2,
        latent_dim=2,
        xvec_max_audio_seconds=10.0,
        dit=SimpleNamespace(hidden_size=4),
        vocoder=SimpleNamespace(sample_rate=48_000, hop_size=2),
    )
    components = SimpleNamespace(
        layout=SimpleNamespace(
            config=config,
            qwen_config=SimpleNamespace(max_position_embeddings=4_096),
        ),
        core=core,
        audio_vae=_AudioVAE(),
        latent_io=_LatentIO(),
    )
    return DotsTTSGenerator(components, tokenizer), backend, dit


def test_cached_generator_compacts_full_history_after_cache_publication() -> None:
    generator, _, _ = _generator("meanflow")

    class CompactSolver:
        unit_length = 3
        hidden_patch_size = 1

        def __init__(self):
            self.full_sequences = []
            self.tails = []

        def sample(self, state, *, sequence, **kwargs):
            del kwargs
            self.full_sequences.append(sequence)
            state.cache = object()
            return mx.full((1, 2, 2), 0.25)

        def sample_tail(
            self,
            state,
            *,
            previous_unit,
            current_hidden,
            **kwargs,
        ):
            del state, kwargs
            self.tails.append((previous_unit, current_hidden))
            return mx.full((1, 2, 2), 0.5)

    solver = CompactSolver()
    current = mx.full((1, 1, 4), 0.7)
    cfg_current = mx.full((1, 1, 4), -0.7)
    state = _DotsTTSRequestState(
        fm_chunks=[
            mx.full((1, 3, 4), 0.1),
            mx.full((1, 3, 4), 0.4),
            current,
        ],
        cfg_chunks=[
            mx.full((1, 3, 4), -0.1),
            mx.full((1, 3, 4), -0.4),
            cfg_current,
        ],
        qwen_cache=None,
        semantic_state=None,
        dit_solver=solver,
        dit_state=SimpleNamespace(cache=None),
        rng=SimpleNamespace(next_key=lambda: mx.random.key(123)),
    )

    first_patch = generator._solve_patch(
        state,
        speaker_condition=None,
        solver_steps=1,
        guidance_scale=1.2,
    )
    assert len(solver.full_sequences) == 1
    assert int(solver.full_sequences[0].shape[1]) == 9
    assert [int(chunk.shape[1]) for chunk in state.fm_chunks] == [1]
    assert [int(chunk.shape[1]) for chunk in state.cfg_chunks] == [1]
    np.testing.assert_array_equal(state.fm_chunks[0], current)
    np.testing.assert_array_equal(state.cfg_chunks[0], cfg_current)

    generator._append_history(first_patch, state.fm_chunks, state.cfg_chunks)
    next_hidden = mx.full((1, 1, 4), 0.9)
    generator._append_hidden(next_hidden, state.fm_chunks, state.cfg_chunks)
    expected_previous = mx.concatenate(state.fm_chunks[:-1], axis=1)
    expected_current = state.fm_chunks[-1]

    generator._solve_patch(
        state,
        speaker_condition=None,
        solver_steps=1,
        guidance_scale=1.2,
    )
    assert len(solver.tails) == 1
    np.testing.assert_array_equal(solver.tails[0][0], expected_previous)
    np.testing.assert_array_equal(solver.tails[0][1], expected_current)
    assert [int(chunk.shape[1]) for chunk in state.fm_chunks] == [1]
    assert [int(chunk.shape[1]) for chunk in state.cfg_chunks] == [1]


@pytest.mark.parametrize("mode", ["flow_matching", "meanflow"])
def test_solvers_are_deterministic_finite_non_silent_and_budgeted(mode: str) -> None:
    generator, _, dit = _generator(mode)
    first = generator.synthesize(
        "A",
        max_audio_patches=3,
        solver_steps=2,
        seed=7,
        eos_threshold=1.0,
    )
    second = generator.synthesize(
        "A",
        max_audio_patches=3,
        solver_steps=2,
        seed=7,
        eos_threshold=1.0,
    )
    np.testing.assert_array_equal(
        np.asarray(first.waveform), np.asarray(second.waveform)
    )
    assert first.sample_rate == 48_000
    assert first.waveform.ndim == 1
    assert first.num_patches == 3
    assert bool(mx.all(mx.isfinite(first.waveform)).item())
    assert bool(mx.any(mx.abs(first.waveform) > 0).item())
    assert dit.calls == 2 * 3 * 2


def test_streaming_decodes_first_patches_then_merged_groups_and_residual() -> None:
    generator, _, _ = _generator("flow_matching")
    chunks = list(
        generator.synthesize_stream(
            "A",
            max_audio_patches=8,
            solver_steps=1,
            seed=7,
            eos_threshold=1.0,
            stream_chunk_patches=4,
        )
    )

    assert generator.components.audio_vae.decode_calls == [
        (2, False),
        (2, False),
        (8, False),
        (4, False),
        (0, True),
    ]
    assert [chunk.num_patches for chunk in chunks] == [1, 1, 4, 2]
    assert all(chunk.waveform.ndim == 1 for chunk in chunks)
    assert all(chunk.waveform.dtype == mx.float32 for chunk in chunks)
    assert all(int(chunk.waveform.size) > 0 for chunk in chunks)


def test_synthesize_batch_decodes_once_and_preserves_low_level_patch_metadata() -> None:
    generator, _, _ = _generator("flow_matching")
    streamed = list(
        generator.synthesize_stream(
            "A",
            max_audio_patches=7,
            solver_steps=1,
            seed=11,
            eos_threshold=1.0,
        )
    )
    generator.components.audio_vae.decode_calls.clear()
    aggregate = generator.synthesize(
        "A",
        max_audio_patches=7,
        solver_steps=1,
        seed=11,
        eos_threshold=1.0,
    )

    assert isinstance(aggregate, DotsTTSSynthesisOutput)
    np.testing.assert_array_equal(
        aggregate.waveform,
        mx.concatenate([chunk.waveform for chunk in streamed]),
    )
    assert aggregate.sample_rate == 48_000
    assert aggregate.num_patches == 7
    assert sum(chunk.num_patches for chunk in streamed) == aggregate.num_patches
    assert generator.components.audio_vae.decode_calls == []
    assert len(generator.components.audio_vae.batch_latents) == 1
    assert int(generator.components.audio_vae.batch_latents[0].shape[-1]) == 14


@pytest.mark.parametrize("mode", ["flow_matching", "meanflow"])
@pytest.mark.parametrize(
    ("activation_dtype", "atol", "rtol"),
    (
        (mx.float32, 0.0, 0.0),
        (mx.bfloat16, 1e-2, 1e-2),
    ),
)
def test_batch_and_streaming_share_seeded_payload_latents(
    mode: str,
    activation_dtype: mx.Dtype,
    atol: float,
    rtol: float,
) -> None:
    batch_generator, _, _ = _generator(mode, activation_dtype)
    batch = batch_generator.synthesize(
        "A",
        max_audio_patches=8,
        solver_steps=1,
        seed=19,
        eos_threshold=1.0,
    )
    stream_generator, _, _ = _generator(mode, activation_dtype)
    chunks = list(
        stream_generator.synthesize_stream(
            "A",
            max_audio_patches=8,
            solver_steps=1,
            seed=19,
            eos_threshold=1.0,
        )
    )

    batch_latent = batch_generator.components.audio_vae.batch_latents[0]
    stream_latent = mx.concatenate(
        [
            latent
            for latent in stream_generator.components.audio_vae.stream_latents
            if int(latent.shape[-1])
        ],
        axis=-1,
    )
    streamed_waveform = mx.concatenate([chunk.waveform for chunk in chunks])
    mx.eval(batch_latent, stream_latent, batch.waveform, streamed_waveform)

    np.testing.assert_array_equal(
        np.asarray(batch_latent.astype(mx.float32)),
        np.asarray(stream_latent.astype(mx.float32)),
    )
    assert batch.num_patches == sum(chunk.num_patches for chunk in chunks) == 8
    assert int(batch_latent.shape[-1]) == 8 * batch_generator.config.patch_size
    assert int(batch.waveform.size) == int(streamed_waveform.size)
    np.testing.assert_allclose(
        streamed_waveform,
        batch.waveform,
        atol=atol,
        rtol=rtol,
    )
    seams = np.cumsum([int(chunk.waveform.size) for chunk in chunks])[:-1]
    for seam in seams:
        start = max(0, int(seam) - 2)
        end = min(int(batch.waveform.size), int(seam) + 2)
        np.testing.assert_allclose(
            streamed_waveform[start:end],
            batch.waveform[start:end],
            atol=atol,
            rtol=rtol,
        )


def test_same_seeded_streams_match_standalone_when_interleaved(monkeypatch) -> None:
    original_normal = mx.random.normal

    def keyed_normal(*args, **kwargs):
        assert kwargs.get("key") is not None
        return original_normal(*args, **kwargs)

    monkeypatch.setattr(mx.random, "normal", keyed_normal)

    def new_stream(generator):
        return generator.synthesize_stream(
            "A",
            max_audio_patches=8,
            solver_steps=1,
            seed=29,
            eos_threshold=1.0,
            stream_chunk_patches=3,
        )

    standalone = []
    for _ in range(2):
        generator, _, _ = _generator("flow_matching")
        standalone.append(
            [np.asarray(chunk.waveform) for chunk in new_stream(generator)]
        )

    shared_generator, _, _ = _generator("flow_matching")
    streams = [new_stream(shared_generator), new_stream(shared_generator)]
    interleaved: list[list[np.ndarray]] = [[], []]
    active = [True, True]
    while any(active):
        for index, stream in enumerate(streams):
            if not active[index]:
                continue
            try:
                interleaved[index].append(np.asarray(next(stream).waveform))
            except StopIteration:
                active[index] = False

    for expected_chunks, actual_chunks in zip(
        standalone,
        interleaved,
        strict=True,
    ):
        assert len(actual_chunks) == len(expected_chunks)
        for expected, actual in zip(expected_chunks, actual_chunks, strict=True):
            np.testing.assert_array_equal(actual, expected)


def _assert_request_state_released(state) -> None:
    assert state.fm_chunks == []
    assert state.cfg_chunks == []
    assert state.qwen_cache is None
    assert state.semantic_state is None
    assert state.dit_solver is None
    assert state.dit_state is None
    assert state.rng is None


def test_request_state_is_released_after_normal_completion(monkeypatch) -> None:
    generator, _, _ = _generator("flow_matching")
    states = []
    original_prefill = generator._prefill

    def capture_state(schedule, prompt, state):
        states.append(state)
        return original_prefill(schedule, prompt, state)

    monkeypatch.setattr(generator, "_prefill", capture_state)
    generator.synthesize(
        "A",
        max_audio_patches=3,
        solver_steps=1,
        eos_threshold=1.0,
    )

    assert len(states) == 1
    _assert_request_state_released(states[0])


def test_request_state_is_released_after_generation_failure(monkeypatch) -> None:
    generator, _, _ = _generator("flow_matching")
    states = []
    original_prefill = generator._prefill

    def capture_state(schedule, prompt, state):
        states.append(state)
        return original_prefill(schedule, prompt, state)

    def fail_solver(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("solver failed")

    monkeypatch.setattr(generator, "_prefill", capture_state)
    monkeypatch.setattr(generator, "_solve_patch", fail_solver)
    with pytest.raises(RuntimeError, match="solver failed"):
        generator.synthesize(
            "A",
            max_audio_patches=3,
            solver_steps=1,
            eos_threshold=1.0,
        )

    assert len(states) == 1
    _assert_request_state_released(states[0])


def test_early_stream_close_does_not_flush_or_retain_request_state(
    monkeypatch,
) -> None:
    generator, _, dit = _generator("flow_matching")
    states = []
    original_prefill = generator._prefill

    def capture_state(schedule, prompt, state):
        states.append(state)
        return original_prefill(schedule, prompt, state)

    monkeypatch.setattr(generator, "_prefill", capture_state)
    stream = generator.synthesize_stream(
        "A",
        max_audio_patches=6,
        solver_steps=1,
        eos_threshold=1.0,
    )
    first = next(stream)
    stream.close()

    assert first.num_patches == 1
    assert generator.components.audio_vae.decode_calls == [(2, False)]
    assert dit.calls == 1
    assert not hasattr(generator, "request_state")
    assert len(states) == 1
    _assert_request_state_released(states[0])


def test_patch_budget_cap_and_explicit_test_double_solver_path() -> None:
    generator, _, dit = _generator("flow_matching")
    assert DEFAULT_MAX_AUDIO_PATCHES == 500
    with pytest.raises(ValueError, match="must not exceed 512"):
        generator.synthesize("A", max_audio_patches=513)

    solver, state = generator._new_dit_request(1)
    assert solver is state is None
    dit._dots_tts_full_history_test_double = False
    with pytest.raises(AttributeError):
        generator._new_dit_request(1)


def test_continuation_speaker_only_and_no_reference_schedule_semantics(
    monkeypatch,
) -> None:
    generator, backend, _ = _generator("flow_matching")
    speaker = mx.ones((1, 4))
    continuation = DotsTTSPromptConditioning(
        speaker,
        mx.ones((1, 1, 2, 2)),
        mx.ones((1, 2, 2)),
    )
    monkeypatch.setattr(
        generator, "prepare_prompt", lambda *args, **kwargs: continuation
    )
    output = generator.synthesize(
        "new",
        reference_audio=mx.ones((8,)),
        reference_text="ref",
        language="en",
        max_audio_patches=3,
        solver_steps=1,
        eos_threshold=1.0,
    )
    assert output.num_patches == 1
    assert "[EN]ref\nnew" in backend.encoded

    speaker_only = DotsTTSPromptConditioning(speaker, None, None)
    monkeypatch.setattr(
        generator, "prepare_prompt", lambda *args, **kwargs: speaker_only
    )
    generator.synthesize(
        "new",
        reference_audio=mx.ones((8,)),
        max_audio_patches=1,
        solver_steps=1,
        eos_threshold=1.0,
        language="en",
    )
    assert "[EN]new" in backend.encoded

    no_reference = DotsTTSPromptConditioning(None, None, None)
    monkeypatch.setattr(
        generator, "prepare_prompt", lambda *args, **kwargs: no_reference
    )
    generator.synthesize(
        "match",
        max_audio_patches=1,
        solver_steps=1,
        eos_threshold=1.0,
    )
    assert "match" in backend.encoded


def test_continuation_semantic_dtype_preserves_vocoder_payload(
    monkeypatch,
) -> None:
    generator, _, _ = _generator("flow_matching", mx.bfloat16)
    continuation = DotsTTSPromptConditioning(
        mx.ones((1, 4), dtype=mx.bfloat16),
        mx.ones((1, 1, 2, 2), dtype=mx.bfloat16),
        mx.ones((1, 2, 2), dtype=mx.bfloat16),
    )
    monkeypatch.setattr(
        generator, "prepare_prompt", lambda *args, **kwargs: continuation
    )

    semantic = generator.components.core.semantic_encoder
    semantic_inputs = []
    original_prefill = semantic.prefill
    original_decode_patch = semantic.decode_patch

    def record_prefill(value, *, max_audio_patches=None):
        semantic_inputs.append(("prefill", value))
        return original_prefill(value, max_audio_patches=max_audio_patches)

    def record_decode_patch(value, state):
        semantic_inputs.append(("decode", value))
        return original_decode_patch(value, state)

    monkeypatch.setattr(semantic, "prefill", record_prefill)
    monkeypatch.setattr(semantic, "decode_patch", record_decode_patch)

    denormalized = []

    def fp32_denormalize(value):
        payload = value.astype(mx.float32) + mx.array(0.25, dtype=mx.float32)
        denormalized.append(payload)
        return payload

    monkeypatch.setattr(generator.components.latent_io, "denormalize", fp32_denormalize)
    payloads = list(
        generator._generate_latent_patches(
            "new",
            reference_audio=mx.ones((8,)),
            reference_text="ref",
            max_audio_patches=3,
            solver_steps=1,
            eos_threshold=1.0,
        )
    )
    mx.eval(*payloads, *denormalized)

    assert semantic_inputs[0][0] == "prefill"
    assert semantic_inputs[0][1].dtype == mx.bfloat16
    generated_semantic_inputs = [
        value for operation, value in semantic_inputs if operation == "decode"
    ]
    assert len(generated_semantic_inputs) == 2
    assert all(value.dtype == mx.bfloat16 for value in generated_semantic_inputs)
    assert len(payloads) == 1
    assert payloads[0].dtype == mx.float32
    np.testing.assert_array_equal(payloads[0], denormalized[-1])


def test_prepare_prompt_only_encodes_latents_for_transcript_backed_reference(
    monkeypatch,
) -> None:
    generator, _, _ = _generator("flow_matching")
    waveform = mx.ones((8,))
    calls = []
    monkeypatch.setattr(generator, "_load_reference", lambda *args, **kwargs: waveform)
    monkeypatch.setattr(
        generator,
        "_speaker_condition",
        lambda *args, **kwargs: mx.ones((1, 4)),
    )

    def prompt_latents(value, **kwargs):
        assert kwargs["key"].shape == (2,)
        calls.append(value)
        return mx.ones((1, 1, 2, 2)), mx.ones((1, 2, 2))

    monkeypatch.setattr(generator, "_prompt_latents", prompt_latents)
    speaker_only = generator.prepare_prompt(
        waveform,
        reference_text=None,
    )
    assert speaker_only.prompt_patches is None
    assert not calls
    continuation = generator.prepare_prompt(
        waveform,
        reference_text="transcript",
    )
    assert continuation.prompt_patch_count == 1
    assert len(calls) == 1
    with pytest.raises(ValueError, match="requires reference_audio"):
        generator.prepare_prompt(None, reference_text="transcript")


@pytest.mark.parametrize(
    ("source_rate", "expected_length", "indices", "expected", "total", "energy"),
    [
        (
            16_000,
            291,
            [0, 1, 2, 97, 145, 194, 288, 289, 290],
            [
                0.0615330264,
                0.09110618383,
                0.1174066141,
                -0.0149249183,
                0.1085735261,
                0.0705845058,
                0.06263947487,
                0.05769650638,
                0.03223892301,
            ],
            1.47331221722,
            6.936226697,
        ),
        (
            24_000,
            194,
            [0, 1, 2, 64, 97, 129, 191, 192, 193],
            [
                0.06106276438,
                0.0954708755,
                0.1199210733,
                0.2327937931,
                -0.003141458379,
                -0.1406152695,
                0.08313312382,
                0.0788994357,
                0.04228758439,
            ],
            1.36746536824,
            4.58672763317,
        ),
        (
            44_100,
            106,
            [0, 1, 2, 35, 53, 70, 103, 104, 105],
            [
                0.06064910814,
                0.09632117301,
                0.1179325432,
                -0.2670885324,
                0.08094123751,
                0.02591892332,
                0.2529423833,
                0.3379882574,
                0.1641791165,
            ],
            1.80146456393,
            2.43650386923,
        ),
    ],
)
def test_high_quality_reference_resampler_matches_pinned_kaiser_oracles(
    source_rate,
    expected_length,
    indices,
    expected,
    total,
    energy,
) -> None:
    # Pinned from the independent vectorized form of the official torchaudio
    # sinc_interp_kaiser kernel (width=64, rolloff=0.95, default Kaiser beta).
    time = np.arange(97, dtype=np.float32) / source_rate
    waveform = (
        0.21 * np.sin(2 * np.pi * 997 * time + 0.13)
        + 0.07 * np.cos(2 * np.pi * 2311 * time - 0.4)
        + np.linspace(-0.03, 0.04, 97, dtype=np.float32)
    ).astype(np.float32)
    first = _high_quality_resample(waveform, source_rate, 48_000)
    second = _high_quality_resample(waveform, source_rate, 48_000)
    assert first.shape == (expected_length,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[indices], expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(first.sum(dtype=np.float64), total, atol=3e-6)
    np.testing.assert_allclose(np.sum(first.astype(np.float64) ** 2), energy, atol=3e-6)


@pytest.mark.parametrize("source_rate", [8_000, 22_050, 32_000, 96_000, 192_000])
def test_high_quality_reference_resampler_covers_standard_rates(source_rate) -> None:
    waveform = np.linspace(-0.25, 0.4, 113, dtype=np.float32)
    first = _high_quality_resample(waveform, source_rate, 48_000)
    second = _high_quality_resample(waveform, source_rate, 48_000)
    assert first.shape == ((waveform.size * 48_000 + source_rate - 1) // source_rate,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def _dense_output_oracle(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    output_length = (waveform.size * target_rate + source_rate - 1) // source_rate
    output = np.empty(output_length, dtype=np.float64)
    sample_indices = np.arange(waveform.size, dtype=np.float64)
    scale = 0.95 * min(1.0, target_rate / source_rate)
    denominator = np.i0(14.769656459379492)
    for output_index in range(output_length):
        source_position = output_index * source_rate / target_rate
        time = np.clip(
            (sample_indices - source_position) * scale,
            -64.0,
            64.0,
        )
        inside = np.clip(1.0 - (time / 64.0) ** 2, 0.0, None)
        window = np.i0(14.769656459379492 * np.sqrt(inside)) / denominator
        coefficients = (np.sinc(time) * window * scale).astype(np.float32)
        output[output_index] = np.sum(
            waveform * coefficients,
            dtype=np.float32,
        )
    return output.astype(np.float32)


@pytest.mark.parametrize("source_rate", [44_117, 47_999, 48_001])
def test_coprime_resampling_is_bounded_and_matches_dense_oracle(source_rate) -> None:
    waveform = np.array(
        [0.17, -0.08, 0.31, -0.22, 0.05, 0.11, -0.04],
        dtype=np.float32,
    )
    plan = _resample_plan(waveform.size, source_rate, 48_000)
    first = _high_quality_resample(waveform, source_rate, 48_000)
    second = _high_quality_resample(waveform, source_rate, 48_000)
    assert (
        plan.output_length == (waveform.size * 48_000 + source_rate - 1) // source_rate
    )
    assert plan.workspace_bytes <= _RESAMPLE_WORKSPACE_BYTES
    assert first.shape == (plan.output_length,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first,
        _dense_output_oracle(waveform, source_rate, 48_000),
        atol=3e-6,
        rtol=3e-6,
    )


def test_exact_rate_resampling_is_deterministic_float32() -> None:
    waveform = np.linspace(-1.0, 1.0, 31, dtype=np.float64)
    expected = waveform.astype(np.float32)
    first = _high_quality_resample(waveform, 48_000, 48_000)
    second = _high_quality_resample(waveform, 48_000, 48_000)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("source_rate", "target_rate"),
    [
        (0, 48_000),
        (-1, 48_000),
        (True, 48_000),
        (16_000.0, 48_000),
        (48_000, 0),
        (48_000, False),
        (48_000, 48_000.5),
    ],
)
def test_resampling_rejects_invalid_rates_before_allocation(
    monkeypatch,
    source_rate,
    target_rate,
) -> None:
    waveform = np.ones(4, dtype=np.float32)
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated"))
    monkeypatch.setattr(np, "arange", lambda *args, **kwargs: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="positive integer sample rate"):
        _high_quality_resample(waveform, source_rate, target_rate)


def test_resampling_rejects_impossible_workspace_before_allocation(monkeypatch) -> None:
    waveform = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated"))
    monkeypatch.setattr(np, "arange", lambda *args, **kwargs: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="cannot fit one output row"):
        _high_quality_resample(waveform, 10_000_000, 1)


def test_resampling_rejects_unsupported_integer_range_before_allocation(
    monkeypatch,
) -> None:
    waveform = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated"))
    monkeypatch.setattr(np, "arange", lambda *args, **kwargs: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="integer index range"):
        _high_quality_resample(
            waveform,
            np.iinfo(np.int64).max + 1,
            np.iinfo(np.int64).max,
        )


def test_resampling_rejects_excessive_output_before_allocation(monkeypatch) -> None:
    waveform = np.broadcast_to(np.ones(1, dtype=np.float32), (70_000_000,))
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated"))
    monkeypatch.setattr(np, "arange", lambda *args, **kwargs: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="bounded allocation limit"):
        _high_quality_resample(waveform, 48_000, 48_000)


def test_path_and_memory_references_share_high_quality_resampling(monkeypatch) -> None:
    generator, _, _ = _generator("flow_matching")
    waveform = mx.linspace(-0.2, 0.3, 41)
    monkeypatch.setattr(
        "mlx_speech.audio.load_audio",
        lambda path, *, mono: (waveform, 16_000),
    )
    from_path = generator._load_reference(
        "reference.wav",
        reference_sample_rate=None,
    )
    from_memory = generator._load_reference(
        waveform,
        reference_sample_rate=16_000,
    )
    np.testing.assert_array_equal(np.asarray(from_path), np.asarray(from_memory))


def test_speaker_only_resamples_only_the_configured_prefix(monkeypatch) -> None:
    generator, _, _ = _generator("flow_matching")
    generator.speaker_frontend.max_audio_seconds = 0.01
    waveform = mx.linspace(-0.2, 0.3, 4_000)
    calls = []
    original = _high_quality_resample

    def tracked_resample(value, source_rate, target_rate, *, max_output_samples=None):
        calls.append((value.size, max_output_samples))
        return original(
            value,
            source_rate,
            target_rate,
            max_output_samples=max_output_samples,
        )

    monkeypatch.setattr(
        "mlx_speech.generation.dots_tts._high_quality_resample",
        tracked_resample,
    )
    monkeypatch.setattr(
        generator,
        "_speaker_condition",
        lambda value, **kwargs: mx.ones((1, 4)),
    )
    prompt = generator.prepare_prompt(
        waveform,
        reference_text=None,
        reference_sample_rate=192_000,
    )
    assert prompt.prompt_patches is None
    assert calls == [(2_187, 480)]
    source = np.asarray(waveform, dtype=np.float32)
    full = original(source, 192_000, 48_000)
    limited = original(source[:2_187], 192_000, 48_000, max_output_samples=480)
    np.testing.assert_array_equal(limited, full[:480])


def test_prompt_budget_rejects_before_audio_encoder_and_leaves_speaker_only_unchanged(
    monkeypatch,
) -> None:
    generator, _, dit = _generator("flow_matching")
    chunk = generator.config.patch_size * generator.config.vocoder.hop_size
    waveform = mx.ones((5 * chunk,))
    calls = []
    monkeypatch.setattr(generator, "_load_reference", lambda *args, **kwargs: waveform)
    monkeypatch.setattr(
        generator,
        "_speaker_condition",
        lambda *args, **kwargs: mx.ones((1, 4)),
    )

    def guarded_encoder(value):
        calls.append(value)
        raise AssertionError("AudioVAE prompt encoder must not run")

    monkeypatch.setattr(generator, "_prompt_latents", guarded_encoder)
    with pytest.raises(ValueError, match="regenerated prompt tail"):
        generator.synthesize(
            "payload",
            reference_audio=waveform,
            reference_text="transcript",
            max_audio_patches=5,
            solver_steps=1,
        )
    assert not calls
    assert generator.components.core.qwen.steps == 0
    assert dit.calls == 0

    output = generator.synthesize(
        "speaker only",
        reference_audio=waveform,
        max_audio_patches=1,
        solver_steps=1,
        eos_threshold=1.0,
    )
    assert output.num_patches == 1
    assert not calls


def test_over_budget_continuation_rejects_before_resampling_or_conditioning(
    monkeypatch,
) -> None:
    generator, _, _ = _generator("flow_matching")
    waveform = mx.ones((20,))
    calls = []
    monkeypatch.setattr(
        "mlx_speech.generation.dots_tts._high_quality_resample",
        lambda *args, **kwargs: calls.append("resample"),
    )
    monkeypatch.setattr(
        generator,
        "_speaker_condition",
        lambda *args, **kwargs: calls.append("speaker"),
    )
    monkeypatch.setattr(
        generator,
        "_prompt_latents",
        lambda *args, **kwargs: calls.append("audio_vae"),
    )
    with pytest.raises(ValueError, match="regenerated prompt tail"):
        generator.prepare_prompt(
            waveform,
            reference_text="transcript",
            reference_sample_rate=44_100,
            max_audio_patches=5,
        )
    assert not calls


def test_too_short_continuation_rejects_before_resampling_or_conditioning(
    monkeypatch,
) -> None:
    generator, _, _ = _generator("flow_matching")
    waveform = mx.ones((1,))
    calls = []
    monkeypatch.setattr(
        "mlx_speech.generation.dots_tts._high_quality_resample",
        lambda *args, **kwargs: calls.append("resample"),
    )
    monkeypatch.setattr(
        generator,
        "_speaker_condition",
        lambda *args, **kwargs: calls.append("speaker"),
    )
    monkeypatch.setattr(
        generator,
        "_prompt_latents",
        lambda *args, **kwargs: calls.append("audio_vae"),
    )
    with pytest.raises(ValueError, match="too short"):
        generator.prepare_prompt(
            waveform,
            reference_text="transcript",
            reference_sample_rate=44_100,
            max_audio_patches=5,
        )
    assert not calls


def test_too_short_prompt_rejects_before_audio_encoder(monkeypatch) -> None:
    generator, _, _ = _generator("flow_matching")
    chunk = generator.config.patch_size * generator.config.vocoder.hop_size
    waveform = mx.ones((chunk,))
    calls = []
    monkeypatch.setattr(generator, "_load_reference", lambda *args, **kwargs: waveform)
    monkeypatch.setattr(
        generator,
        "_prompt_latents",
        lambda value: calls.append(value),
    )
    with pytest.raises(ValueError, match="too short"):
        generator.synthesize(
            "payload",
            reference_audio=waveform,
            reference_text="transcript",
            max_audio_patches=3,
        )
    assert not calls


def test_schedule_position_limit_rejects_before_qwen_or_solver() -> None:
    generator, _, dit = _generator("flow_matching")
    generator.components.layout.qwen_config.max_position_embeddings = 8
    with pytest.raises(ValueError, match="max_position_embeddings"):
        generator.synthesize(
            "oversized schedule",
            max_audio_patches=20,
            solver_steps=1,
        )
    assert generator.components.core.qwen.steps == 0
    assert dit.calls == 0


def test_continuation_suppresses_first_eos_and_requires_payload_budget(
    monkeypatch,
) -> None:
    generator, _, dit = _generator("flow_matching")
    generator.components.core.qwen.stop = True
    prompt = DotsTTSPromptConditioning(
        mx.ones((1, 4)),
        mx.ones((1, 1, 2, 2)),
        mx.ones((1, 2, 2)),
    )
    monkeypatch.setattr(generator, "prepare_prompt", lambda *args, **kwargs: prompt)
    output = generator.synthesize(
        "payload",
        reference_audio=mx.ones((8,)),
        reference_text="prompt",
        max_audio_patches=4,
        solver_steps=1,
    )
    assert output.num_patches == 1
    assert dit.calls == 2  # two solved patches, with SOAR branches batched per call
    with pytest.raises(ValueError, match="no payload"):
        generator.synthesize(
            "payload",
            reference_audio=mx.ones((8,)),
            reference_text="prompt",
            max_audio_patches=2,
            solver_steps=1,
        )


def test_fm_mask_keeps_history_causal_and_latent_block_fully_visible() -> None:
    mask = np.asarray(_build_fm_attention_mask(5, 2))[0]
    np.testing.assert_array_equal(mask[:4, :4], np.tri(4, dtype=bool))
    assert mask[4].all()
    assert mask[5:].all()
