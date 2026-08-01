# dots.tts Default-Path Inference Efficiency Plan

## Goal

Implement the bounded outcome in [SPEC.md](SPEC.md): reduce cached 128-patch MF and SOAR generation time by at least 35% without quality, determinism, streaming, memory, or checkpoint regressions.

## Architecture Approach

Execution follows [DESIGN.md](DESIGN.md): one request-local latent-patch producer feeds either a one-shot batch decoder or the bounded 1/1/4 streaming decoder; model-owned compiled/packed helpers never own request state; DiT storage grows transactionally and uses unpublished tail storage as attention scratch; prompt reuse is bounded and content-aware.

The plan deliberately measures the current cached path once, then optimizes the largest measured stages. It does not restore a repeated uncached benchmark or publish performance claims.

## Ordered Slice Sequence

### Slice 1: Reconcile in-flight work and freeze canonical comparison evidence

**Objective:** Establish the exact cached starting point and make both performance and quality comparison inputs durable before runtime optimization begins.

**Acceptance criteria:**
- Retain the in-flight compact DiT tail only after the existing MF/SOAR oracle tests prove full/compact parity, request isolation, BF16 behavior, and delayed publication.
- Replace `benchmark_dots_tts_cache.py` and its relative-speed tests with `profile_dots_tts_inference.py` and focused tests. The profiler has no backend selector and performs no uncached timing.
- Add a tested comparison-contract helper shared by the profiler and quality runner. It populates and validates the single JSON block in the linked Slice detail, preserves unrelated Markdown, and fails closed on incomplete cases, incompatible schemas, or identity mismatches. The quality runner also gains repeated `--case` selection for the focused Slice 3 decision; omitting it still runs the complete final matrix.
- Freeze complete performance inputs—source and working-tree identity, MLX/host and artifact/reference digests, workload, all measured trials, batch total/RTF, streaming TTFC/completion, stage totals, patch count, output health, compile/warmup time, and peak memory—in canonical Slice evidence.
- Import the existing fixed quality report into the same canonical contract with its report, manifest, corpus, ASR, artifact, threshold, and per-case metric data. If the ignored source report is absent, regenerate it once on the unchanged starting tree before any decoder change.
- Remove the unaccepted cache-streaming performance report. Raw JSON under `outputs/dots_tts/inference_efficiency/` remains diagnostic and is not an input required by later slices.
- Freeze `batch` and `stream` cases for each base MF/SOAR artifact using the approved fixed input and seed. Each case has one compilation warmup, then clears only prompt features before exactly three measured requests; request one is cold-reference and requests two/three are same-reference warm.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_inference_profile.py tests/unit/test_dots_tts_quant_gate.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch stream --warmup-runs 1 --runs 3 --max-audio-patches 128 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/baseline.json --freeze-comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md
.venv/bin/python scripts/eval/dots_tts_quant_gate.py --freeze-comparison-report outputs/dots_tts/quant_gate/report.json --comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md
```

**Touches:** current compact-tail edits, `scripts/eval/`, profiler/quality-runner unit tests, stale untracked benchmark/report files

**Produces:** an adopted cached starting tree, raw local diagnostics, and a self-contained canonical comparison contract

**Detail:** [Canonical comparison contract and evidence schema](slices/slice-001.md)

### Slice 2: Share latent generation across batch and streaming decode sinks

**Objective:** Make ordinary generation batch-decode once while streaming retains bounded 1/1/4 delivery over the same acoustic generation core.

**Acceptance criteria:**
- Factor prompt/schedule/EOS/DiT/semantic/Qwen/RNG processing into one internal latent-patch producer with request-local cleanup.
- `DotsTTSGenerator.synthesize()` collects payload latents and calls `AudioVAE.decode()` once; `synthesize_stream()` alone owns streaming merge, recurrent state, lookahead flush, and early close.
- `DotsTTSAdapter.generate()` calls the non-streaming synthesis path directly; public arguments, `TTSOutput`, `num_patches`, the 500 default, and the 512 maximum remain compatible.
- Batch and streaming paths use identical seeded latent patches, payload counts, sample counts, and dtype-aware waveform/seam tolerances for MF and SOAR test doubles.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_adapter.py tests/unit/test_dots_tts_vocoder_streaming.py
```

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `generation/dots_tts.py`, dots.tts adapter, generation/adapter/streaming tests

**Produces:** one acoustic producer with distinct latency and streaming decode sinks

**Status:** complete

**Evidence:** [Slice 2 execution summary](slices/slice-002-summary.md)

### Slice 3: Correct decoder precision and compile the vocoder step

**Objective:** Keep BigVGAN in its checkpoint dtype and reuse compiled common-shape SLSTM/decoder execution without sharing request state.

**Acceptance criteria:**
- Add real and synthetic assertions for the dtype entering `BigVGANDecoder`; the rolling decoder window matches `decoder.conv_pre.weight.dtype` for base and int8 artifacts.
- Attempt BF16 post-projection and SLSTM execution first, then run real-checkpoint batch/stream waveform and seam checks plus a four-case fixed quality gate against the canonical Slice 1 contract in this slice. The selected cases cover MF/SOAR, continuation/speaker-only, and English/Chinese on base artifacts; int8 entry dtype remains covered structurally because its decoder weights follow the same BF16 policy. If any tolerance fails, retain only SLSTM accumulation in FP32, cast once before BigVGAN, and rerun the failing checks before Slice 4 may begin.
- Cache pure-tensor compiled functions for common one-patch/four-patch chunk shapes and observed residual shapes. Keys include operation, shape, dtype, and relevant model identity; request RNG/state is passed explicitly.
- Warm calls reuse compiled functions, flush does not rerun recurrence, early close remains lazy, and eager/compiled state plus waveform outputs match within dtype-aware tolerances.
- Record the accepted precision boundary and all quality deltas in Slice evidence. Record a one-run 32-patch MF stage profile after the slice; compilation is retained only when warm execution is no slower than the equivalent correct eager path.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_audio_vae.py tests/unit/test_dots_tts_vocoder.py tests/unit/test_dots_tts_vocoder_streaming.py tests/unit/test_dots_tts_checkpoint_contract.py
.venv/bin/python -m pytest tests/runtime/test_dots_tts_base.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf --paths batch stream --warmup-runs 1 --runs 1 --max-audio-patches 32 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/slice-3.json
.venv/bin/python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts --peak-memory-limit-gib 30 --force --case mf/base/samantha_en_us/continuation --case mf/base/tingting_zh_cn/speaker_only --case soar/base/samantha_en_us/speaker_only --case soar/base/tingting_zh_cn/continuation --output-dir outputs/dots_tts/inference_efficiency/slice-3-quality --report outputs/dots_tts/inference_efficiency/slice-3-quality.md --comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md
```

**Execution:** subagent recommended

**Depends on:** Slice 2

**Touches:** AudioVAE, BigVGAN, checkpoint dtype assertions, runtime batch/stream coverage, quality gate, profiler stage evidence

**Produces:** precision-correct warm-reused vocoder execution

### Slice 4: Grow DiT cache storage on demand

**Objective:** Separate request maximum from physical cache capacity and make 64/128/256/512 growth transactional.

**Acceptance criteria:**
- `DiTSolverState` retains the 512-bounded request maximum but allocates bucket 64 first and resolves later capacity from current finalized history.
- Growth allocates replacement K/V, copies only published tokens for every NFE/layer/branch, materializes the copy, and swaps state only after success.
- Exact K/V, offsets, dtypes, branch layout, and solver outputs survive 64→128→256→512 transitions; injected allocation/copy failures leave the prior cache usable.
- A default 500-patch request that emits two patches proves it allocated 64 rather than 512. Unit tests correct the prior one-patch maximum-smoke assumption.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit_cache.py -k 'bucket or grow or capacity or failure or request'
```

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `models/dots_tts/dit_inference.py`, request creation, DiT cache tests

**Produces:** demand-sized request-local DiT storage with rollback-safe transitions

### Slice 5: Remove full-prefix K/V concatenation

**Objective:** Attend over contiguous published history plus fresh scratch tail without copying the complete DiT prefix in every layer and NFE.

**Acceptance criteria:**
- Fresh previous-unit/current-unit K/V is slice-written after the published offset and read through one contiguous view; the per-layer `mx.concatenate(cached_prefix, fresh)` path is absent.
- Request offsets advance by exactly one five-token finalized unit only after all NFE evaluations succeed; current hidden/noisy-tail K/V never becomes published.
- Existing compact sequence-tail handling is retained without storing full `fm_chunks`/`cfg_chunks` after cache publication requires only the fixed fresh tail.
- MF/SOAR first, later, continuation, CFG, BF16, interleaved-request, and injected mid-NFE failure oracles match the trusted solver and inspect exact cache content.
- Record one 32-patch cached MF/SOAR stage profile; DiT stage time and allocation must not regress from the Slice 4 path.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_generation.py -k 'dit or cache or compact or interleav or failure or continuation'
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch --warmup-runs 1 --runs 1 --max-audio-patches 32 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/slice-5.json
```

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** DiT runner/cache publication, generator history retention, oracle tests

**Produces:** bounded later-patch attention without full-prefix K/V copies

### Slice 6: Pack and reuse DiT inference kernels

**Objective:** Remove checkpoint-shaped projection overhead from stable DiT hot steps while preserving serialization and request isolation.

**Acceptance criteria:**
- Build inference-only packed QKV and block/final adaptive-modulation projections from loaded modules without registering new serialized parameters or changing conversion/checkpoint contracts.
- Cache solver/runner helpers by mode, bucket, dtype, branch count, and model identity; request conditioning, RNG, cache arrays, and offsets remain explicit inputs/state.
- First-patch, prompt-prefill, and later-patch eager-packed outputs match the unpacked oracle. Compiled variants are enabled individually only when warm stage timing improves and parity passes.
- Base/int8 loading remains exact, interleaved requests cannot observe another request's conditioning/cache, and compile cost is reported separately.
- Record one 32-patch cached MF/SOAR stage profile after packing/compilation.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_checkpoint.py tests/unit/test_dots_tts_checkpoint_contract.py tests/checkpoint/test_dots_tts_base_load.py tests/checkpoint/test_dots_tts_int8_load.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch --warmup-runs 1 --runs 1 --max-audio-patches 32 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/slice-6.json
```

**Execution:** subagent recommended

**Depends on:** Slice 5

**Touches:** DiT modules/inference helpers, solver reuse, checkpoint/parity tests

**Produces:** faster reusable DiT inference kernels with unchanged artifacts

### Slice 7: Reuse prompt features safely

**Objective:** Reduce repeated-reference TTFC with a bounded content-aware cache that preserves scale and seed semantics.

**Acceptance criteria:**
- Add a 256-entry memory-only LRU holding unscaled speaker embeddings and eligible pre-sampling prompt latent distributions keyed by normalized audio content and prompt mode.
- Changed files or arrays miss by content; speaker scale and request seed do not alias cached projected conditions or sampled latents.
- Lookup/insertion is lock-protected without holding the lock during speaker/AudioVAE execution; duplicate concurrent computation is safe and eviction is deterministic.
- Same-reference calls skip eligible encoders, while continuation/speaker-only, MF/SOAR, base/int8, different scale, different seed, and interleaved requests preserve outputs and request isolation.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_speaker.py tests/unit/test_dots_tts_audio_vae.py -k 'prompt or reference or speaker or cache or seed or interleav'
```

**Depends on:** Slice 6

**Touches:** prompt preparation/cache, generator lifetime, prompt and concurrency tests

**Produces:** deterministic warm-reference conditioning reuse

### Slice 8: Remove redundant synchronization and mode-dead work

**Objective:** Minimize host/device boundaries and residual orchestration work after the dominant vocoder and DiT paths are fixed.

**Acceptance criteria:**
- Remove the redundant outer patch evaluation, combine waveform materialization with finite/non-silent reductions, and allow the waveform boundary to publish recurrent decoder outputs where transactional safety permits.
- Preserve the necessary EOS control-flow boundary, pre-yield invalid-audio failure, cache failure atomicity, early-close behavior, and deterministic interleaving; add instrumentation assertions for `mx.eval`/`.item()` counts.
- MeanFlow creates no CFG chunks/projections; SOAR reuses constant unconditional projection and fixed masks/positions. Retained generator history remains fixed-tail bounded after DiT cache publication.
- Run a post-cleanup 32-patch MF/SOAR stage profile that separates Qwen and semantic time for the residual-stage decision in Slice 9.
- Update `docs/dots-tts.md` for the separate batch/stream sinks, prompt reuse, progressive cache memory, compilation warmup, 1/1/4 streaming cadence, 500/512 bounds, and current long-text limitation; make no performance claim.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_dit_cache.py -k 'cache or eval or sync or meanflow or cfg or interleav or incremental'
.venv/bin/python -m pytest tests/unit/test_dots_tts_release.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch --warmup-runs 1 --runs 1 --max-audio-patches 32 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/slice-8.json
.venv/bin/ruff check src scripts tests
git diff --check
```

**Depends on:** Slice 7

**Touches:** generation synchronization, mode-specific history, request lifetime tests, `docs/dots-tts.md`

**Produces:** bounded low-synchronization orchestration, a residual-stage profile, and accurate runtime documentation

### Slice 9: Close only material Qwen and semantic residual overhead

**Objective:** Apply fixed-shape Qwen/semantic reuse only when the post-cleanup profile shows it remains material to total latency.

**Acceptance criteria:**
- Use `slice-8.json` as the decision input. Qwen plus semantic work is material when their combined share exceeds 15% of total time or either individual share exceeds 10% for MF or SOAR.
- If neither threshold is crossed, record the measured no-change decision in Slice evidence and leave shared Qwen/semantic runtime code untouched.
- If a threshold is crossed, add request-independent fixed-shape workspace/compile reuse for the measured stage only; prefill remains batched, valid K/V lengths remain explicit, and request-owned caches are function inputs/outputs rather than compiled captures.
- Any shared Qwen change preserves VibeVoice cache dtype, tuple compatibility, incremental/full parity, and interleaved request behavior. Semantic changes preserve full/chunk parity, mask/rotary positions, and capacity failure behavior.
- Run the same 32-patch MF/SOAR profile after any implementation and require the targeted warm stage to improve without total-time regression.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_qwen.py tests/unit/test_dots_tts_semantic_encoder.py tests/unit/test_vibevoice_qwen2.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch --warmup-runs 1 --runs 1 --max-audio-patches 32 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/slice-9.json
```

**Execution:** subagent recommended

**Depends on:** Slice 8

**Touches:** profile evidence and, only when triggered, shared Qwen or dots.tts semantic inference helpers/tests

**Produces:** an evidence-backed residual-stage implementation or explicit no-change closure

### Slice 10: Prove performance, memory, quality, and repository gates

**Objective:** Verify the completed default path against the canonical comparison contract and every behavioral invariant without introducing new implementation, test, evaluator, or documentation changes.

**Acceptance criteria:**
- The `batch` case for each MF/SOAR base artifact improves median ordinary-generation time by at least 35% versus the canonical Slice 1 contract. The separate `stream` case records cold/warm TTFC and streaming completion time without replacing the batch-total gate. Both cases use one compilation warmup and three cached 128-patch requests.
- Two-patch 512-budget smokes for MF and SOAR cross first cache publication and remain below 30 GiB; allocation evidence proves bucket 64 is physical while 512 remains only the request maximum.
- Unit, checkpoint, runtime, local base integration, batch/stream waveform parity, deterministic/interleaved generation, and release checks pass once at the final gate.
- Regenerated MF/SOAR × base/int8 × continuation/speaker-only evidence passes the existing int8 gate and stays within WER `0.01` and speaker-cosine `0.02` regression of the canonical Slice 1 quality data.
- Both evaluators validate all canonical identities before comparison; gitignored raw baselines may be absent. This slice records final evidence only and makes no source, test, evaluator, documentation, performance-report, model-card, or remote mutation.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_dots_tts_base.py
.venv/bin/python scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch stream --warmup-runs 1 --runs 3 --max-audio-patches 128 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md --minimum-batch-improvement 0.35 --maximum-bucket-smoke-patches 2 --output outputs/dots_tts/inference_efficiency/final.json
.venv/bin/python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts --peak-memory-limit-gib 30 --force --output-dir outputs/dots_tts/inference_efficiency/quality --report outputs/dots_tts/inference_efficiency/quality.md --comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md
.venv/bin/ruff check src scripts tests
.venv/bin/python scripts/hugging_face/upload.py dots-tts --dry-run
git diff --check
```

**Depends on:** Slice 9

**Touches:** verification evidence only

**Produces:** final cached-path performance, memory, quality, and repository acceptance evidence

## Requirement Traceability

| Requirement / gap | Satisfying slices |
| --- | --- |
| EFF-001 / GAP-DEC-01 | Slices 2, 10 |
| EFF-002 / GAP-DEC-02 / GAP-DEC-03 | Slices 1, 3, 10 |
| EFF-003 / GAP-DIT-01 | Slices 4, 10 |
| EFF-004 / GAP-DIT-02 / GAP-DIT-03 | Slices 1, 5, 6, 10 |
| EFF-005 / GAP-PROMPT-01 | Slices 7, 10 |
| EFF-006 / GAP-SYNC-01 / GAP-AUX-01 | Slices 8, 9, 10 |
| EFF-007 evidence contract | Slices 1, 3, 5, 6, 8, 9, 10 |

## Execution Routing and Topology

Default: direct, serial, and continuous after each slice verifies.

Overrides:
- Slice 2: subagent recommended because it changes the internal generation interface across the generator and public adapter.
- Slice 3: subagent recommended because dtype and compiled recurrent/DSP execution can create quality or seam regressions without structural failure.
- Slices 4–6: subagent recommended because progressive allocation, unpublished scratch, projection packing, and compiled runners share delayed-commit invariants.
- Slice 9: subagent recommended because a residual Qwen optimization, if triggered, affects the shared VibeVoice path.

**Parallel-safe groups:** none. Every performance decision and baseline comparison depends on the verified output of the preceding slice.

Checkpoints: none. Continue through all approved slices; execution windows are context-management boundaries, not planned stops.

## Aggregate Verification Commands

| Gate | Command |
| --- | --- |
| Canonical baselines | Slice 1 profiler and quality-runner freeze commands populate and validate `slices/slice-001.md`; ignored JSON is diagnostic only |
| Batch/stream architecture | focused generation, adapter, and vocoder-streaming unit tests |
| Vocoder precision/compile | AudioVAE/vocoder/dtype tests, real runtime waveform coverage, Slice 3 stage profile, and early fixed quality comparison |
| DiT storage/publication | cache transition, oracle, failure, checkpoint, and Slice 5/6 stage profiles |
| Prompt/synchronization | focused prompt, interleaving, eval-boundary, Qwen/semantic/VibeVoice tests |
| Final repository | `pytest tests/unit/ tests/checkpoint/ tests/runtime/`, local base integration, Ruff, release dry run, and `git diff --check` |
| Final performance/memory | cached-only final profiler with the canonical contract, 35% gate, and two-patch maximum-bucket smokes |
| Final quality | fixed multilingual quant gate compared directly with the canonical Slice 1 quality contract |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The revised plan makes both comparison gates self-contained, settles decoder precision before downstream optimization, and leaves final verification free of new implementation work.
- Concern: Slice 5 writes fresh K/V into unpublished cache storage before lazy MLX evaluation completes, so an incorrect alias or slice bound could corrupt published history even when offsets do not advance.
- Action: Require Slice 5's injected mid-NFE test to snapshot and exactly compare the published prefix and offsets before failure, after failure, and after retry before committing the slice.
- Verified: Traced the current batch and stream decoder flow, AudioVAE precision boundaries, selective int8 policy, DiT cache mutation, quality-runner matrix, canonical-contract lifecycle, slice dependencies, and final verification commands.
