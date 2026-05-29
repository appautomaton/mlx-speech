# Granite Speech Long-Audio Memory Plan

## Goal

Implement `.agent/work/2026-05-29-granite-speech-long-audio-memory/SPEC.md`: fix Granite Speech runtime memory behavior, add coarse memory telemetry, and rerun `/tmp` long-audio checks with efficiency and accuracy reporting.

## Architecture Approach

Use `.agent/work/2026-05-29-granite-speech-long-audio-memory/DESIGN.md`. The key runtime change is replacing the manual Granite LM attention implementation with MLX efficient scaled-dot-product attention, preserving GQA without KV pre-tiling, and moving audio prompt context validation before encoder/projector work.

## Execution Routing And Topology

- Default execution: direct.
- Continue through all slices after each slice verifies.
- Parallel-safe groups: none. Later slices depend on the corrected runtime memory behavior.
- Checkpoints: none.
- Review recommendation: run `auto-eng-review` before execution because the attention/cache change is numerically sensitive and affects runtime correctness.

## Ordered Slice Sequence

### Slice 1: Efficient Granite LM Attention And Cache Dtype

**Objective:** Replace hand-materialized Granite LM attention with MLX efficient attention and make KV cache dtype explicit and bounded.

**Acceptance criteria:**

- Granite attention uses `mx.fast.scaled_dot_product_attention` or an equivalent MLX efficient primitive.
- The implementation no longer manually materializes `scores` and `weights` as `[B, heads, L, L]` arrays.
- GQA KV heads are not pre-tiled/repeated before attention.
- KV cache allocation follows the active input/model dtype instead of unconditional fp32.
- Tiny LM tests still cover prefill, cached decode, cache length increments, and cache dtype.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py tests/unit/test_granite_speech_memory_bounds.py && rg -n "scaled_dot_product_attention" src/mlx_speech/models/granite_speech_asr/language_model.py && ! rg -n "scores = mx\\.matmul|weights = mx\\.softmax|_repeat_kv\\(" src/mlx_speech/models/granite_speech_asr/language_model.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/language_model.py`, unit tests.

**Status:** complete
**Evidence:** replaced manual Granite LM attention with `mx.fast.scaled_dot_product_attention`, removed GQA KV pre-tiling, allocated KV cache from runtime/model dtype, and added cache dtype coverage; `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py tests/unit/test_granite_speech_memory_bounds.py` passed (9 tests); source guard found `scaled_dot_product_attention` and no old `scores`/`weights`/`_repeat_kv` patterns.
**Risks / next:** efficient attention semantics are shape-tested but final transcript behavior remains covered by later smoke and long-audio benchmark slices.

### Slice 2: Early Context Preflight And Runtime Cleanup

**Objective:** Reject over-context audio requests before heavy model work and avoid retaining unnecessary large arrays after transcription.

**Acceptance criteria:**

- `GraniteSpeechAsrModel.transcribe(...)` computes prompt token count from audio sample count before STFT/encoder/projector allocation.
- An over-context request raises `ValueError` before `get_audio_features(...)` or encoder/projector execution.
- Normal transcription still runs feature extraction, audio projection, mixed embedding prefill, and greedy decode.
- Runtime does not retain per-step logits or hidden-state history after return.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_generation.py tests/unit/test_granite_speech_memory_bounds.py`

**Touches:** `src/mlx_speech/generation/granite_speech_asr.py`, unit tests.

**Status:** complete
**Evidence:** moved Granite prompt/context validation to sample-count preflight before feature extraction and audio projection, with mismatch protection between preflight and extraction token counts; `.venv/bin/python -m pytest tests/unit/test_granite_speech_generation.py tests/unit/test_granite_speech_memory_bounds.py` passed (5 tests), including the over-context path proving feature extraction and `get_audio_features(...)` are not called.
**Risks / next:** normal generation remains unit-proven; full runtime smoke remains in final verification.

### Slice 3: Coarse MLX Memory Telemetry

**Objective:** Add reusable coarse memory snapshots and include them in Granite diagnostic summaries without polling.

**Acceptance criteria:**

- A lightweight helper captures active, cache, and peak MLX memory at named lifecycle boundaries.
- The helper can reset peak memory and clear cache only at explicit boundaries.
- Granite diagnostic summary records memory snapshots and timing fields when telemetry is enabled.
- Unit tests prove telemetry shape and that no per-token/background polling mechanism exists.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_diagnostics.py tests/unit/test_granite_speech_memory_bounds.py`

**Touches:** `scripts/generate/granite_speech_asr.py`, optional helper under `src/mlx_speech/`, unit tests.

**Status:** complete

**Evidence:** added `src/mlx_speech/diagnostics.py` with explicit MLX active/cache/peak snapshots, peak reset, and cache clearing; Granite diagnostic summaries now optionally include timing, token counts, and named memory snapshots at request boundaries; `.venv/bin/python -m pytest tests/unit/test_granite_speech_diagnostics.py tests/unit/test_granite_speech_memory_bounds.py` passed (8 tests).

**Risks / next:** telemetry is summary-shape/unit-proven; real long-audio memory behavior is measured in slice 5.

### Slice 4: `/tmp` Long-Audio Benchmark Driver

**Objective:** Add a repeatable `/tmp`-only benchmark path for public-domain long audio with known script comparison.

**Acceptance criteria:**

- Benchmark driver downloads or reuses public-domain audio and matching text under `/tmp`.
- Driver chunks long audio into context-safe segments without writing media or transcripts under repo `outputs/`.
- Summary records source URLs, chunk count, durations, prompt tokens, generated tokens, wall time, RTF/RTFx, memory snapshots, non-empty status, and normalized word-level accuracy/coverage metrics against the script.
- Unit tests cover chunk planning, summary construction, and text normalization/word metric logic without network or real model weights.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_diagnostics.py tests/unit/test_granite_speech_long_audio_benchmark.py`

**Touches:** `scripts/generate/granite_speech_asr.py` or `scripts/eval/granite_speech_long_audio.py`, unit tests.

### Slice 5: Final Granite Verification And Long-Audio Measurement

**Objective:** Prove the fixed runtime still passes Granite gates and collect post-fix `/tmp` long-audio behavior evidence.

**Acceptance criteria:**

- Existing Granite unit, checkpoint, runtime smoke, and dependency guard tests pass.
- A real `/tmp` long-audio run uses a public-domain >10-minute file with matching script and completes with context-safe chunking.
- Final evidence reports peak MLX memory, active/cache memory after cleanup, duration, wall time, RTF/RTFx, token counts, and word-level accuracy/coverage.
- The report explicitly compares post-fix memory behavior with the prior 100+ GB memory-pressure observation and says whether the issue is resolved or still reproducible.

**Verification:** `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/test_granite_speech_full_load.py tests/runtime/test_granite_speech_smoke.py && tmpdir=$(mktemp -d /tmp/granite-long-audio-fixed.XXXXXX) && .venv/bin/python scripts/eval/granite_speech_long_audio.py --output-dir "$tmpdir" --source three-bears-catamount --chunk-seconds 120 --max-new-tokens 350`

**Touches:** PLAN evidence, docs if behavior notes change.

## Aggregate Verification Commands

| Slice | Command |
| --- | --- |
| 1 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py tests/unit/test_granite_speech_memory_bounds.py && rg -n "scaled_dot_product_attention" src/mlx_speech/models/granite_speech_asr/language_model.py && ! rg -n "scores = mx\\.matmul|weights = mx\\.softmax|_repeat_kv\\(" src/mlx_speech/models/granite_speech_asr/language_model.py` |
| 2 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_generation.py tests/unit/test_granite_speech_memory_bounds.py` |
| 3 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_diagnostics.py tests/unit/test_granite_speech_memory_bounds.py` |
| 4 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_diagnostics.py tests/unit/test_granite_speech_long_audio_benchmark.py` |
| 5 | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/test_granite_speech_full_load.py tests/runtime/test_granite_speech_smoke.py && tmpdir=$(mktemp -d /tmp/granite-long-audio-fixed.XXXXXX) && .venv/bin/python scripts/eval/granite_speech_long_audio.py --output-dir "$tmpdir" --source three-bears-catamount --chunk-seconds 120 --max-new-tokens 350` |

## Engineering Review Recommendation

Run `auto-eng-review` before execution. This plan changes the Granite LM attention primitive, KV cache dtype behavior, and long-audio benchmark path; subtle mistakes could preserve shape tests while degrading transcript quality or cache correctness.
