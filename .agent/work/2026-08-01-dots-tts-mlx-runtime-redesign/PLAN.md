# dots.tts MLX-Native Inference Redesign Plan

## Goal

Implement the bounded outcome in [SPEC.md](SPEC.md): delete the benchmark platform and make the real pure-MLX MF/SOAR batch and streaming paths faster without changing speech behavior.

## Architecture Approach

Execution follows [DESIGN.md](DESIGN.md): stable MLX acoustic graphs with explicit request state, a tiled AudioVAE bridge, bounded stateful BigVGAN decoding, and a thin before/after timer. PyTorch source is read for behavior before each stage is changed; its runtime structure is not copied.

## Execution Routing and Topology

Default: direct, serial, continuation after each verified slice. The user asked the primary agent to finish the work without further agent-team ceremony.

Overrides: none.

**Parallel-safe groups:** none.

Checkpoints: none. The starting measurement runs once in Slice 1 and the final measurement runs once in Slice 6. No privileged capture or human action is required.

## Ordered Slice Sequence

### Slice 1: Remove the benchmark platform and freeze one thin starting measurement

**Objective:** Delete the uncommitted experiment-management machinery, restore an uninstrumented ordinary path, and record one real public-path starting measurement.

**Acceptance criteria:**
- Delete `profile_dots_tts_runtime.py`, `capture_dots_tts_runtime.py`, `dots_tts_runtime_completion_gate.py`, `dots_tts_runtime_evidence.py`, the runtime manifest, registry/ledger artifacts, generalized runtime-profile tests, and private diagnostics used only by that platform.
- Remove platform hooks from generation and AudioVAE while preserving unrelated runtime behavior and the user's `tmp/` contents. Keep `/tmp/` ignored.
- Reduce `profile_dots_tts_inference.py` to a thin default-path timer: public int8 MF/SOAR, normal EOS/default patch bound, batch/stream, one request per cell, separate load, wall, first-audio, waveform duration/RTF, patch count, stop reason when already available, and peak memory.
- The timer exposes no candidate, ledger, capture, backend/reference, comparison-contract, or repetition controls and writes only the requested local JSON file.
- Run the starting command once and retain `/tmp/dots-tts-before.json` as local scratch evidence. Do not rerun it in later slices.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_inference_profile.py tests/unit/test_dots_tts_generation.py
.venv/bin/python -m pytest tests/unit/
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --artifact-class int8 --variants mf soar --paths batch stream --output /tmp/dots-tts-before.json
git diff --check
```

**Depends on:** none
**Touches:** benchmark-platform files, generation diagnostics, legacy dots.tts profiler and tests
**Produces:** clean runtime tree and one non-repeated starting timing file

**Status:** complete
**Evidence:** deleted the uncommitted runtime profiler/capture/ledger/diagnostic platform and restored tracked runtime files to `HEAD`; reduced `profile_dots_tts_inference.py` to one raw request per cell plus a tested `--compare`; focused tests 60 passed and unit suite 826 passed. One starting run wrote `/tmp/dots-tts-before.json`: MF batch/stream 8.847s/14.870s, SOAR batch/stream 12.108s/18.557s.
**Risks / next:** starting timing is intentionally a single raw observation; do not rerun it, and preserve the locked inputs for Slice 6.

### Slice 2: Fix acoustic EOS, cache publication, and host synchronization

**Objective:** Remove avoidable per-patch host work from the Qwen/semantic feedback loop while preserving official stop and rollback semantics.

**Acceptance criteria:**
- Read the pinned PyTorch Qwen/generation source and document any semantic difference in focused test names or short code comments.
- Reuse the Qwen EOS result for the current patch; threshold `1.0` builds neither EOS projection nor scalar publication.
- Normal EOS performs no pre-DiT host read. The current patch is solved, semantically fed back, added to Qwen, emitted, and only then stopped.
- Cache offsets and request state publish only after successful MLX evaluation; injected failure, retry, close, and interleaved requests preserve the previous state.
- No public argument or alternate implementation selector is added.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_qwen.py -k 'eos or stop or cache or rollback or interleav'
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
git diff --check
```

**Depends on:** Slice 1
**Touches:** `src/mlx_speech/generation/dots_tts.py`, Qwen/cache code, focused tests
**Produces:** one retained acoustic feedback path with correct stop ordering

**Status:** complete
**Evidence:** `DotsTTSQwen.step` can skip EOS projection, and generation reuses returned EOS logits, co-materializes the current patch transaction before the scalar decision, yields the patch, then stops. Focused EOS/cache tests 14 passed; the combined tier run completed its 828-unit phase, and isolated dots.tts checkpoint/runtime verification reported 9 passed; scoped Ruff and `git diff --check` passed.
**Risks / next:** none; the one discarded full multi-model tier process lost its final exit status after unit completion, so only the relevant dots.tts higher tiers were rerun.

### Slice 3: Make DiT execution reuse stable MLX work

**Objective:** Replace fragmented and repeated DiT inference work with bounded stable MLX execution without changing MF/SOAR solver math.

**Acceptance criteria:**
- Read the pinned PyTorch DiT/solver source before editing and preserve MF/SOAR schedules, NFE counts, CFG, noise, masks, positions, and cache semantics.
- Implement the smallest reusable compiled/fused boundary that improves the real loop. Compile signatures are bounded by model/mode/dtype/solver/cache capacity, never logical prefix length or request offset.
- Use parity-proven MLX fast normalization, rotary, batched operations, and immutable geometry where they remove repeated eager construction; remove any losing prototype instead of keeping a selector.
- Seeded, cached/full-history, bucket growth, BF16, failure rollback, and interleaved requests match existing fixtures and tolerances.
- Peak MLX memory stays below 30 GiB and compilation cannot grow once per patch.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_solvers.py
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
git diff --check
```

**Depends on:** Slice 2
**Touches:** DiT layers, inference runner/cache, solvers, focused tests
**Produces:** a single bounded-signature DiT implementation

### Slice 4: Compile and vectorize the AudioVAE bridge

**Objective:** Remove avoidable Python/eager recurrent work between acoustic patches and decoder frames.

**Acceptance criteria:**
- Read the pinned PyTorch AudioVAE source and preserve recurrent equations, dtype boundaries, padding, and output layout.
- Replace row-wise fixed projections with batched MLX operations and use a bounded set of reusable tile shapes with tensor valid length.
- Padding never advances hidden/cell state; returned state is exactly after the last valid frame.
- Batch and streaming tiles cover short, steady, residual, and zero-frame finalization without per-length compilation.
- Full and tiled outputs/states match existing dtype-aware tolerances; failures and interleaved requests do not leak state.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_audio_vae.py tests/unit/test_dots_tts_vocoder.py tests/unit/test_dots_tts_vocoder_streaming.py -k 'bridge or state or tile or row or compile or interleav or failure'
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
git diff --check
```

**Depends on:** Slice 3
**Touches:** AudioVAE bridge/recurrent projection code and focused tests
**Produces:** one bounded compiled bridge used by batch and stream

### Slice 5: Eliminate rolling BigVGAN recomputation in streaming

**Objective:** Decode only new frames plus required overlap/lookahead instead of repeatedly decoding the full rolling context.

**Acceptance criteria:**
- Read the pinned PyTorch BigVGAN source and preserve causal padding, transpose overlap, alias-free FIR phase, AMP branch order, lookahead, and final flush.
- Introduce bounded request-owned decoder state and route both batch and streaming through the retained state-correct primitives.
- One-shot and chunked output match for single-frame, regular, mixed, irregular, short, early-final, and duplicate-final partitions; sample counts and seams remain correct.
- State publishes only after successful evaluation; close, failure/retry, and interleaved streams remain isolated.
- Keep existing public streaming arguments. Change steady cadence only if a focused timing shows lower completion time without harming first-audio behavior.
- Use a private Metal kernel only if a specific remaining primitive is measurably material and built-in MLX cannot express it efficiently; otherwise keep the built-in path.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_vocoder.py tests/unit/test_dots_tts_vocoder_streaming.py tests/unit/test_dots_tts_generation.py -k 'stream or chunk or state or seam or flush or close or failure or interleav'
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
git diff --check
```

**Depends on:** Slice 4
**Touches:** BigVGAN primitives, AudioVAE decoder state, batch/stream generator routing, focused tests
**Produces:** one stateful decoder with no rolling full-context recomputation

### Slice 6: Run the single final timing and waveform gate

**Objective:** Verify that the retained runtime is faster on the real public paths and still produces complete, correct speech.

**Acceptance criteria:**
- Run the thin timer once to `/tmp/dots-tts-after.json`; compare the same MF/SOAR batch/stream cells with `/tmp/dots-tts-before.json` without adding trials or rerunning the starting path.
- Every primary cell is faster and none regresses more than 2%. Load, first-audio, waveform duration, RTF, patch count, stop reason, and peak memory remain visible.
- English and Mandarin speaker-only and continuation integration cases produce finite, non-silent 48 kHz waveforms, recover the target tail, and do not exhaust the patch budget.
- Existing WER/speaker, deterministic cloning, interleaving, checkpoint, memory, release, and public API gates pass.
- Repository search confirms no benchmark platform, candidate selector, accepted-head ledger, capture transaction, or measurement-only dependency remains.

**Verification:**
```bash
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --artifact-class int8 --variants mf soar --paths batch stream --output /tmp/dots-tts-after.json --compare /tmp/dots-tts-before.json
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/
.venv/bin/python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts --peak-memory-limit-gib 30 --force --output-dir /tmp/dots-tts-final-quality --report /tmp/dots-tts-final-quality.md
.venv/bin/ruff check src/mlx_speech/generation/dots_tts.py src/mlx_speech/models/dots_tts scripts/eval/profile_dots_tts_inference.py tests/unit
.venv/bin/python scripts/hugging_face/upload.py dots-tts --dry-run
git diff --check
```

**Depends on:** Slice 5
**Touches:** focused completion/quality tests and minimal timing output only
**Produces:** final speed and waveform evidence without a benchmark framework

## Aggregate Verification Commands

| Gate | Command | When |
| --- | --- | --- |
| Default development | `.venv/bin/python -m pytest tests/unit/` | Once at each slice completion |
| Inference/DSP | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/` | Once at completion of Slices 2–6 |
| End-to-end waveform | `RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/` | Slice 6 only |
| Public timing | thin `profile_dots_tts_inference.py` command | Slice 1 before and Slice 6 after only |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The six-slice serial plan removes the benchmark platform, keeps measurement disposable, and orders acoustic, DiT, bridge, and stateful decoder changes behind focused parity gates.
- Concern: Slice 1 can accidentally remove legitimate runtime work while deleting the uncommitted diagnostics platform, so execution must classify the existing diff against `HEAD` before cleanup and preserve every non-platform change.
- Concern: Slice 3 can retain an `mx.compile` boundary that recompiles by logical prefix despite passing numerical tests, so a bounded compile-signature/cardinality test must pass before the slice completes.
- Concern: Slice 5 can introduce ConvTranspose overlap, alias-free FIR phase, lookahead, or final-flush drift, so the default route must not change until one-shot, irregular-partition, duplicate-final, failure-retry, and interleaving tests all pass.
- Concern: Slice 6 depends on the thin timer's `--compare` path even though Slice 1's command does not exercise it, so Slice 1 must add a focused raw before/after comparison test without reviving contracts, trials, or ledgers.
- Action: Execute directly and serially, surface each matching risk before Slices 1, 3, 5, and 6, and keep the original path only in git history rather than behind runtime selectors.
- Verified: Current source and uncommitted diff inspected; cleanup ownership, EOS ordering, DiT cache flow, AudioVAE compiled-window behavior, stateless BigVGAN boundary, verification commands, dependency order, rollback coverage, and absence of privileged capture checkpoints traced against PLAN and DESIGN.
