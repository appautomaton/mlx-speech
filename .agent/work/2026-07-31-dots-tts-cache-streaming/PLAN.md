# dots.tts Bounded Caching and Waveform Streaming Plan

## Goal

Implement the bounded cache and streaming contract in [SPEC.md](./SPEC.md) using the request-local architecture in [DESIGN.md](./DESIGN.md), with no checkpoint or remote artifact mutation.

## Architecture Approach

- Qwen and semantic attention share a project-owned capacity-managed append cache; dots.tts passes known request bounds, while other Qwen consumers may grow in 256-token blocks up to their model limit.
- DiT uses an inference-only delayed-commit cache indexed by NFE, layer, and CFG branch. The uncached solver stays internal for parity and benchmark comparison.
- AudioVAE streaming preserves SLSTM state and re-decodes a mathematically derived bounded BigVGAN window.
- The public surface adds only `StreamingTTSModel` and dots.tts `generate_stream()`. Cache policy remains private, and `generate()` drains the same default stream.

## Ordered Slice Sequence

### Slice 1: Capacity-managed Qwen and semantic K/V

**Objective:** Replace per-step Qwen and semantic K/V concatenation with bounded slice-written state while preserving dots.tts and VibeVoice incremental behavior.

**Acceptance criteria:**
- Add a lightweight internal append cache with keys, values, valid offset, capacity, overflow validation, and valid-prefix fetch.
- Qwen cache growth uses 256-token blocks when an exact capacity is absent and never exceeds `max_position_embeddings`; dots.tts can provide its exact schedule capacity.
- Semantic cache capacity follows the request patch budget while its convolution tail remains fixed-size.
- Full/prefill/incremental parity and projected K/V dtypes remain within existing tolerances.
- VibeVoice shared-Qwen behavior and mixed cache-dtype tests pass.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_bounded_kv_cache.py tests/unit/test_dots_tts_qwen.py tests/unit/test_dots_tts_semantic_encoder.py tests/unit/test_vibevoice_qwen2.py
.venv/bin/ruff check src/mlx_speech/models/_cache.py src/mlx_speech/models/_qwen2.py src/mlx_speech/models/dots_tts/layers.py src/mlx_speech/models/dots_tts/semantic_encoder.py
```

**Execution:** subagent recommended
**Touches:** shared model cache, Qwen2, dots.tts semantic encoder, Qwen/VibeVoice unit tests
**Produces:** bounded append cache and migrated Qwen/semantic state

**Status:** complete
**Evidence:** Added bounded slice-written Qwen/semantic cache state with tuple compatibility, exact/default capacity, collection preflight, transactional offset rollback, and coherent mutable semantic request state; focused Metal suite `20 passed`, full unit suite `746 passed`, Ruff and `git diff --check` passed; spec and quality reviews both `APPROVED`.
**Risks / next:** A rejected append may retain an unused grown allocation beyond the restored offset, but valid prefixes and request-visible state remain unchanged; proceed to per-NFE DiT caching.

### Slice 2: Cached MeanFlow and SOAR DiT runner

**Objective:** Add a numerically checked delayed-commit DiT solver that recomputes only the fresh two-unit tail and reuses finalized history per NFE.

**Acceptance criteria:**
- Add an inference-only DiT runner with 64/128/256/512 buckets and cache layout `[NFE, layer, branch, head, token, head_dim]`.
- MeanFlow uses one branch and SOAR uses separate conditional/unconditional state; cache entries never cross NFE indices.
- First-patch, later-patch, continuation-prefix prefill, positions, masks, speaker conditioning, CFG, and ODE schedules match the full-history oracle.
- Every completed later patch commits exactly the previous five-token unit; the current noisy tail is never persisted.
- Fused QKV helpers, rotary tables, and request modulation/schedule reuse do not alter checkpoint names or loading.
- Cache mutations are materialized at patch boundaries and overflow or invalid alignment raises a clear error without uncached fallback.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_solvers.py tests/unit/test_dots_tts_dit_cache.py
.venv/bin/ruff check src/mlx_speech/models/dots_tts/dit.py src/mlx_speech/models/dots_tts/dit_inference.py src/mlx_speech/models/dots_tts/solvers.py
```

**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** dots.tts DiT, solver, inference cache, focused unit tests
**Produces:** cached DiT solver plus retained internal full-history oracle

**Status:** complete
**Evidence:** Added projected-dtype-aware per-NFE/layer/branch delayed-commit caches, cache-safe mask intersection, batch-aligned SOAR metadata, transactional tail publication, and streamed unpublished prompt prefill; focused Metal suite `30 passed`, full unit suite `766 passed`, Ruff and `git diff --check` passed; spec and quality reviews both `APPROVED`.
**Risks / next:** Only unrecoverable process/device failure during final MLX materialization remains; proceed to bounded stateful AudioVAE decoding.

### Slice 3: Stateful bounded AudioVAE decoding

**Objective:** Replace buffered full-history `decode_chunk` behavior with persistent SLSTM recurrence and a finite-context BigVGAN decoder window.

**Acceptance criteria:**
- SLSTM exposes chunk execution with explicit per-layer hidden/cell state, and its full call delegates from zero state.
- Decoder left context is derived from the local Conv1d, causal transposed-convolution, alias-free activation, and AMPBlock structure rather than a copied constant.
- Vocoder state stores only recurrent state, bounded decoder input, total frames, emitted frames, and maximum chunk size.
- Streaming emits stable samples only, supports partial groups, and flushes final lookahead exactly once on normal completion.
- Chunked SLSTM and concatenated waveform output match trusted full-sequence behavior within dtype-aware tolerances; decoder window size does not grow with utterance length.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_audio_vae.py tests/unit/test_dots_tts_vocoder.py tests/unit/test_dots_tts_vocoder_streaming.py
.venv/bin/ruff check src/mlx_speech/models/dots_tts/audio_vae.py src/mlx_speech/models/dots_tts/vocoder.py
```

**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** dots.tts SLSTM, AudioVAE decoder state, BigVGAN context calculation, unit tests
**Produces:** true incremental waveform decoder

**Status:** complete
**Evidence:** Added explicit per-layer SLSTM recurrence, structurally derived BigVGAN context, a five-field bounded decode state, monotonic stable emission, and idempotent lookahead flush; focused MLX suite `18 passed`, full unit suite `774 passed`, Ruff and `git diff --check` passed; spec and quality reviews both `APPROVED`.
**Risks / next:** Synthetic FP32/BF16 seam coverage passes; real-checkpoint streaming equivalence remains part of Slice 5 integration.

### Slice 4: Request state and optional public streaming API

**Objective:** Integrate all bounded state into one dots.tts iterator and expose waveform chunks without changing other TTS families or the non-streaming result.

**Acceptance criteria:**
- Export `StreamingTTSModel` with `generate_stream(...) -> Iterator[TTSOutput]`; only dots.tts structurally implements it.
- dots.tts streaming mirrors existing clone/generation controls and adds positive `stream_chunk_patches=4`.
- Payload patches 1 and 2 decode separately; later patches use the configured merge size; residual patches and lookahead flush at exhaustion.
- Every yielded waveform is non-empty, one-dimensional float32 at 48 kHz.
- `generate()` drains the same default stream, concatenates chunks, and preserves existing `TTSOutput` behavior.
- The request owns Qwen, semantic, DiT, and vocoder state; early iterator close stops without flush and leaves no model-global state.
- Preserve default `max_audio_patches=500`, reject values above 512, and leave a segment-level internal boundary without implementing sentence splitting.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_adapter.py tests/unit/test_tts_streaming_protocol.py
.venv/bin/ruff check src/mlx_speech/generation/dots_tts.py src/mlx_speech/tts/_adapter.py src/mlx_speech/tts/_adapters/dots_tts.py src/mlx_speech/tts/__init__.py
```

**Execution:** subagent recommended
**Depends on:** Slices 1–3
**Touches:** dots.tts generator, unified TTS protocol/export, dots.tts adapter, generation/API tests
**Produces:** public synchronous waveform streaming and one shared generation core

**Status:** complete
**Evidence:** Added the optional `StreamingTTSModel` protocol, request-local Qwen/semantic/DiT/vocoder/RNG state, dots.tts 1/1/N waveform cadence, early-close handling, and shared streaming/non-streaming generation while preserving `num_patches`; focused MLX suite `45 passed`, full unit suite `781 passed`, Ruff and scoped `git diff --check` passed; spec and quality reviews both `APPROVED` after the request-local RNG correction.
**Risks / next:** Real-checkpoint streaming equivalence and peak-memory behavior remain Slice 5 gates; concurrent workflow-file edits are external to this slice and remain untouched.

### Slice 5: Performance, memory, integration, and quality gates

**Objective:** Prove the cached path meets the accepted twofold speed target, 30 GiB bound, waveform integration, and published cloning-quality tolerances.

**Acceptance criteria:**
- Add `scripts/eval/benchmark_dots_tts_cache.py` with a benchmark-only cached/reference selector and JSON/Markdown output.
- Record model variant, seed, patch count, output seconds, total generation time, time to first chunk, RTF, peak MLX memory, and median speedup.
- With Hank speaker-only reference, seed 42, default NFE, `eos_threshold=1.0`, 128 payload patches, and three measured runs, cached MF and SOAR each reach at least 2.0x median speedup over the internal full-history path.
- Every measured run yields before completion and stays below 30 GiB; a 512-budget one-patch smoke exercises the maximum bucket below the same limit.
- Unit, checkpoint, runtime, local four-artifact clone-mode integration, and the fixed multilingual quality gate pass; WER regression is at most 0.01 and speaker-cosine regression at most 0.02 versus the published report.
- Commit the measured cache/streaming report; keep temporary audio, raw records, and regenerated quality output outside Git.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_dots_tts.py
.venv/bin/python scripts/eval/benchmark_dots_tts_cache.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --runs 3 --max-audio-patches 128 --eos-threshold 1.0 --memory-limit-gib 30 --output-dir /tmp/mlx-speech-dots-cache-benchmark --report docs/benchmarks/dots-tts-cache-streaming-2026-07-31.md
.venv/bin/python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts --peak-memory-limit-gib 30 --force --output-dir /tmp/mlx-speech-dots-cache-quality --report /tmp/mlx-speech-dots-cache-quality.md
```

**Depends on:** Slice 4
**Touches:** cache benchmark runner/tests, local integration, benchmark report, existing quality runner invocation
**Produces:** reproducible PASS evidence for speed, first chunk, memory, waveform, and cloning quality

### Slice 6: Documentation and final repository gate

**Objective:** Publish accurate operational documentation and model-card source from the measured behavior, then run the mandatory repository checks.

**Acceptance criteria:**
- Update `docs/dots-tts.md` with the actual 500 default, 512 maximum, 128 recommendation, optional streaming example, hybrid cadence, merge-size tradeoff, bounded-state behavior, measured results, and lack of automatic sentence splitting.
- Update the docs benchmark index and the local Hugging Face model-card source with only claims supported by Slice 5 evidence.
- Preserve the historical quantization report and all existing artifact hashes/layouts; the dry run selects only the four approved artifacts and card.
- Documentation targets Apple Silicon `mlx-speech` users, uses the existing concise operational reference voice, and makes no universal real-time or upstream-equivalence claim.
- Full unit tests, release-focused tests, Ruff, diff checks, and the non-mutating dots.tts release dry run pass.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/
.venv/bin/python -m pytest tests/unit/test_dots_tts_release.py tests/unit/test_dots_tts_quant_gate.py tests/unit/test_dots_tts_cache_benchmark.py
.venv/bin/ruff check src scripts tests
.venv/bin/python scripts/hugging_face/upload.py dots-tts --dry-run
git diff --check
```

**Depends on:** Slice 5
**Touches:** dots.tts guide, benchmark index/report, model-card source, release and documentation tests
**Produces:** accurate checked documentation and final repository validation

## Requirement Traceability

| Requirement | Satisfying slices |
| --- | --- |
| REQ-001 bounded Qwen/semantic state | Slice 1; Slice 5 regression tiers |
| REQ-002 per-NFE DiT cache | Slice 2; Slice 5 benchmark/integration |
| REQ-003 incremental vocoder state | Slice 3; Slice 4 cadence; Slice 5 waveform gate |
| REQ-004 optional TTS streaming | Slice 4; Slice 6 documentation |
| REQ-005 bounds and compatibility | Slices 1–4; Slice 5 maximum-bucket smoke |
| REQ-006 benchmark and documentation | Slices 5–6 |

## Execution Routing and Topology

Default: direct, serial, and continuous after each slice verifies.

Overrides:
- Slice 1: subagent recommended because it changes shared Qwen cache behavior across dots.tts and VibeVoice.
- Slice 2: subagent recommended because per-NFE delayed commit is a non-obvious solver invariant with silent correctness risk.
- Slice 3: subagent recommended because recurrent and finite-context DSP state can create boundary artifacts without structural failures.
- Slice 4: subagent recommended because it crosses generation state, the common TTS protocol, and adapter compatibility.

**Parallel-safe groups:** none. Slices 2–4 have integration dependencies even where their initial write sets appear separate.

Checkpoints: none. Continue through all verified slices, then continue inline into `auto-verify`.

## Aggregate Verification Commands

| Gate | Command |
| --- | --- |
| Shared cache parity | `pytest` Slice 1 focused Qwen/Semantic/VibeVoice tests |
| DiT parity | `pytest` Slice 2 DiT/solver/cache tests |
| Vocoder parity | `pytest` Slice 3 AudioVAE/vocoder streaming tests |
| Public streaming | `pytest` Slice 4 generation/adapter/protocol tests |
| Full runtime | `pytest tests/unit/ tests/checkpoint/ tests/runtime/` plus local dots.tts integration |
| Performance and memory | `benchmark_dots_tts_cache.py` with three 128-patch runs and 30 GiB cap |
| Cloning quality | `dots_tts_quant_gate.py` against the fixed multilingual corpus |
| Final repository | full unit tier, Ruff, release dry run, and `git diff --check` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The six-slice order isolates shared cache, DiT, vocoder, API integration, measured gates, and documentation with explicit parity and rollback-safe code-only verification.
- Concern: Slice 1 changes tuple-shaped Qwen cache objects that are re-exported by VibeVoice and directly unpacked by current tests, so every caller must migrate atomically or retain compatible iteration semantics.
- Concern: Slice 2 is the riskiest slice because an NFE, CFG-branch, mask, or five-token delayed-commit offset error can produce plausible but mathematically incorrect audio unless the multi-patch oracle tests inspect cache contents as well as outputs.
- Concern: Slice 3 can pass coarse waveform tolerances while introducing chunk-seam artifacts if the derived BigVGAN window mishandles replicated alias-free padding or lookahead boundaries.
- Concern: Slice 4 must preserve the exported low-level `DotsTTSSynthesisOutput.num_patches` contract, which runtime and integration tests exercise even though the new unified streaming chunks reuse `TTSOutput`.
- Concern: Slice 5 must reset the seed and isolate request state for every cached/reference trial or the twofold comparison will measure different noise trajectories and invalid peak-memory carryover.
- Action: Require cache-compatibility, per-NFE content/offset, seam-sample, `num_patches`, and per-trial seed/state-reset assertions in the named slice tests before accepting their verification evidence.
- Verified: Traced Qwen cache construction and VibeVoice exports, semantic append state, full-history DiT generation, AudioVAE buffered decode, low-level synthesis metadata, slice dependencies, failure behavior, and all planned verification commands against the current source tree.
