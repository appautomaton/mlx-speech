# dots.tts Default-Path Inference Efficiency

**Supersedes:** [dots.tts Bounded Caching and Waveform Streaming](../2026-07-31-dots-tts-cache-streaming/SPEC.md)

**Bet:** Removing avoidable precision promotion, repeated decoder work, oversized cache allocation, and unfused hot-loop execution will make the default cached dots.tts path materially faster without trading away speech quality, deterministic cloning, bounded memory, or optional streaming.

## Bounded Goal

Make the default MLX dots.tts inference path at least 35% faster than its cached starting baseline for both MeanFlow and SOAR on the same Apple host, while preserving output quality and public behavior, keeping streaming responsive, and replacing benchmark-oriented work with implementation changes that reduce actual TTFC, RTF, and memory cost.

## Classification

- Work scale: capability
- Work shape: mixed — performance refactor, upstream execution parity, and regression coverage
- Selected lenses: product, engineering, runtime

## Target Stakeholder

Apple Silicon developers using dots.tts through `mlx-speech`, especially applications that keep a model loaded, reuse a reference voice, and care about how long a real request takes rather than how favorable a cached-versus-uncached benchmark looks.

## Broader Intent and Scope Preservation

This change preserves the full intent of the parked cache-streaming work: request-local bounded state, deterministic seeded generation, optional waveform streaming, a 500-patch public default, and a 512-patch hard maximum remain supported. Completed implementation from that change is the starting point, not work to be rolled back.

The prior requirements that ordinary `generate()` drain the waveform stream and that completion prove a twofold speedup against an uncached reference are intentionally replaced. They conflict with the user's priority of absolute default-path inference efficiency. The old unfinished plan remains parked as historical context; it is not the execution contract for this change.

## Evidence and Approved Direction

The pinned upstream checkout is `5ed719e3`. It establishes the intended scheduling and optimized execution mechanisms, but PyTorch/CUDA implementation choices are reference evidence rather than runtime dependencies.

- The official streaming cadence is the first two payload patches separately, then groups of four. Local scheduling matches it; changing that default is not the performance strategy.
- Upstream ordinary generation collects latents and decodes once, while local ordinary generation drains repeated fixed-window streaming decode. The shared latent-generation logic should remain single-source, but non-streaming and streaming callers may use different decode sinks.
- Local profiling of the cached path attributed roughly 39 of 54 seconds to AudioVAE/BigVGAN decoding for a 128-patch MeanFlow request. This establishes priority, not a universal timing claim.
- Decoder checkpoint weights are BF16, while local decoder inputs and rolling state are created as FP32. Normal MLX promotion is a high-confidence risk that requires one instrumented confirmation before relying on it as measured fact.
- Local DiT state resolves the request's maximum budget immediately; the normal 500-patch default therefore selects a 512-patch cache even for an early-EOS request. Upstream grows through 64/128/256/512 buckets.
- The existing 512-budget one-patch smoke never reached DiT cache allocation and does not prove maximum-bucket peak memory. A valid smoke must cross the first cache publication boundary.
- Upstream reuses prompt speaker features and prompt latent distributions; local prompt preparation recomputes them for every request.

The normative gap map and closure evidence are defined in [spec/performance-gaps.md](spec/performance-gaps.md).

## Required Outcome

### EFF-001 — Efficient default and streaming decode sinks

- Ordinary `generate()` and low-level non-streaming synthesis use one latency-oriented batch AudioVAE decode after shared latent generation rather than paying the 1/1/4 streaming-window schedule internally.
- `generate_stream()` retains the official 1/1/4 cadence, bounded state, early-close behavior, and non-empty 48 kHz float32 chunks.
- The latent-generation, EOS, conditioning, and patch-count logic remains shared; the optimization must not create two divergent acoustic-generation algorithms.
- Concatenated default streaming output remains within the existing dtype-aware waveform and seam tolerances of batch output.

### EFF-002 — Precision-correct compiled vocoder execution

- BigVGAN consumes and retains decoder-window activations in its BF16 checkpoint dtype unless an explicitly documented FP32 boundary is required for numerical stability.
- Decoder-side SLSTM recurrence preserves the minimum precision needed by quality and parity evidence, then crosses into BigVGAN at the decoder weight dtype.
- Common first-patch, four-patch, residual, and flush shapes reuse compiled or equivalently fused MLX execution instead of rebuilding the full Python-unrolled decoder graph for every chunk.
- Compilation is cached by the model/shape/dtype signature and is not repeated per request.

### EFF-003 — Demand-sized DiT cache storage

- The public maximum remains 512 patches, but physical DiT K/V storage starts at the smallest official bucket required by current finalized history and grows through 64, 128, 256, and 512 only when crossed.
- Bucket transitions copy only published K/V and preserve every NFE, layer, branch, dtype, position, and offset invariant.
- A failed transition or failed solver evaluation does not publish partial state.
- The default 500-patch budget no longer forces 512-bucket allocation on short requests.

### EFF-004 — Bounded DiT hot-loop work

- Inference-only packed QKV and adaptive-modulation projections may be derived from loaded checkpoint modules without changing serialized names or conversion output.
- Stable first-patch, continuation-prefill, and later-patch work reuses compiled or fused MLX execution where it improves measured time without capturing request-owned mutable state incorrectly.
- Later-patch attention does not concatenate and copy the complete valid K/V prefix in every layer and NFE. Fresh tail K/V is written into unpublished cache storage or an equivalent contiguous bounded workspace, becomes visible only after the solver step succeeds, and never publishes the noisy latent tail.
- The existing compact fixed-tail sequence work remains eligible only if oracle parity and absolute cached-path timing support it.

### EFF-005 — Reused prompt conditioning

- A bounded in-process cache reuses normalized reference-audio speaker features and, when continuation text is present, the pre-sampling AudioVAE latent distribution.
- Cache keys account for normalized audio content and every setting that changes the reusable result. Mutable input arrays and changed files cannot return stale conditioning.
- Speaker scaling and request-local latent sampling remain outside cached values so different scales and seeds preserve current semantics.
- Cache capacity and eviction are deterministic, thread-safe at the generator's supported concurrency level, and do not serialize or persist user audio.

### EFF-006 — Reduced synchronization and secondary hot-path overhead

- Redundant `mx.eval` and scalar `.item()` boundaries are removed or combined without weakening cache publication, invalid-audio detection, early iterator close, or exception timing guarantees.
- MeanFlow does not compute or retain unused CFG history. SOAR constant unconditional projections and fixed masks/positions are reused when safe.
- One-token Qwen and one-patch semantic decoding reuse stable workspaces or compiled shapes only after higher-cost vocoder and DiT gaps are addressed and measurements show remaining value.

### EFF-007 — Honest local performance evidence

- Before implementation, record one reproducible cached starting baseline from the exact starting tree. The baseline records its commit plus working-tree identity because compact-tail work is currently in flight.
- On the same host and model artifact, use identical speaker-only reference input, text, seed 42, default solver steps, `eos_threshold=1.0`, and exactly 128 payload patches. Checkpoint loading and one-time compilation warmup are excluded from steady-state timing and reported separately.
- Each baseline and completion case uses one disclosed warmup followed by exactly three measured cached requests on the same loaded model instance. Cold TTFC is measured with an empty prompt-feature cache; warm TTFC is measured on the immediately repeated reference.
- The completion gate compares only cached default paths: the median total generation time for both MeanFlow and SOAR is at least 35% lower than the frozen starting baseline.
- Record absolute total time, output duration, RTF, cold and same-reference warm TTFC, stage times, compile/warmup time, and peak MLX memory. These are local acceptance artifacts, not publication claims.
- Uncached/full-history execution remains a focused correctness oracle and is not run repeatedly as a performance denominator.

## Acceptance Criteria

1. For MF and SOAR, ordinary generation uses the batch decode sink, streaming retains 1/1/4 cadence, both produce the same patch count and sample count, and waveform/seam differences stay within existing dtype-aware tolerances.
2. Instrumented real-checkpoint coverage proves the dtype entering BigVGAN matches its BF16 decoder weight dtype; quality evidence determines any narrower FP32 recurrence boundary.
3. Warm repeated calls hit compiled decoder shapes rather than recompiling, and first-call compile cost is reported separately from steady-state TTFC and RTF.
4. Cache-allocation tests show short default-budget requests allocate bucket 64 first, transitions preserve exact valid K/V, and both MF and SOAR pass two-patch 512-budget memory smokes below 30 GiB peak MLX memory.
5. Tiny deterministic MF and SOAR oracles prove fused/compiled DiT execution and contiguous cache-tail writes match the trusted full-history solver for first patch, later patches, continuation prefill, CFG, speaker conditioning, and injected failure rollback.
6. Same-reference repeated requests skip speaker encoding and eligible prompt AudioVAE encoding while different audio, changed files, scale, transcript mode, and seeds retain their documented semantics.
7. Evaluation-boundary tests show no redundant returned-patch evaluation, combined waveform health reductions, correct pre-yield failure behavior, and early iterator close without hidden flush work.
8. MF avoids unused CFG graph/state growth; retained request history and cache storage remain bounded through 512 patches for both modes.
9. Against the frozen cached starting baseline, median 128-patch total generation time improves by at least 35% for both MF and SOAR. The report includes absolute RTF, cold/warm TTFC, stage timing, compilation cost, and peak memory without repeated uncached trials.
10. Existing MF/SOAR × base/int8 × continuation/speaker-only quality coverage stays within absolute WER regression `0.01` and speaker-cosine regression `0.02`; seeded and interleaved request determinism remain intact.
11. `pytest tests/unit/`, the relevant checkpoint/runtime tiers, real-checkpoint waveform integration, and the fixed quality gate pass on Metal before completion.

## Constraints and Risks

- Runtime remains pure MLX. No Torch, torchaudio, Transformers, `mlx-lm`, `mlx-audio`, or upstream package may become a runtime dependency.
- Checkpoint schemas, weight artifacts, public model aliases, conversion output, and the selective int8 policy do not change.
- MLX compilation must not capture request-local RNG, offsets, recurrent state, or mutable caches across concurrent/interleaved requests.
- BF16 decoder execution can expose seam or quality regressions even when global waveform error looks small; seam-local and released quality gates are mandatory.
- A batch decode sink improves non-streaming completion time but can raise transient memory versus bounded streaming. It must stay under the 30 GiB gate and retain streaming as the bounded-memory option.
- Progressive cache growth can create transition latency spikes. Stage evidence must distinguish allocation/transition time from steady later-patch time.
- Prompt-feature reuse creates invalidation and privacy risk. The cache is bounded, memory-only, content-addressed, and stores derived tensors rather than serialized audio.
- Hardware timing is noisy. Fixed inputs, identical artifacts, a warmup disclosure, and median cached trials are required; thresholds must not be rescued by slowing a reference path.

## Scope Coverage

### Included

- Separate batch and streaming AudioVAE decode sinks over shared latent generation.
- Decoder dtype correction, common-shape compilation/fusion, and bounded streaming parity.
- Progressive DiT buckets, contiguous cache-tail publication, inference-only projection fusion, and measured compile opportunities.
- Bounded prompt-feature reuse, synchronization cleanup, MeanFlow CFG removal, and evidence-backed Qwen/semantic hot-step work.
- Focused correctness oracles, absolute cached-path performance evidence, memory validation, quality gates, and documentation of actual runtime behavior.

### Deferred / Not in scope

- Text-token-arrival or upstream-style double streaming. It improves interactive producer-to-audio TTFC but not fixed-text model compute.
- Automatic sentence splitting, cross-segment waveform stitching, async serving, live playback, or a new streaming CLI.
- K/V quantization, paged attention, rotating eviction, cache serialization, or cross-process prompt caches. These introduce distinct quality, API, or privacy decisions.
- Remote Hugging Face changes, model-card publication, weight uploads, or a publishable benchmark report. Local measurements exist only to accept or reject the implementation.

## Anti-goals

- Do not change the official default 1/1/4 streaming cadence merely to improve non-streaming throughput.
- Do not reduce solver steps, patch count, context, model precision outside the justified decoder boundary, or speech quality to satisfy the speed gate.
- Do not inflate speedup by making an uncached/reference path slower or by comparing different noise trajectories, outputs, artifacts, or request setup.
- Do not duplicate latent-generation, EOS, prompt-continuation, or schedule logic between batch and streaming APIs.
- Do not expose cache buckets, compiled runners, solver state, cache toggles, or benchmark-only controls through the public TTS API.
- Do not silently fall back to full-history DiT execution after an optimized-path error.
- Do not broaden this change into general serving infrastructure or optimization of unrelated model families.
