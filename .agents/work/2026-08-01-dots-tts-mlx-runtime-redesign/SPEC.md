# dots.tts MLX-Native Inference Redesign

**Supersedes:** [dots.tts Default-Path Inference Efficiency](../2026-07-31-dots-tts-inference-efficiency/SPEC.md) and the benchmark-platform framing previously recorded for this active change.

**Bet:** Reimplementing the actual dots.tts inference path around MLX execution semantics—not building more measurement infrastructure—will remove the host-driven work, repeated decoding, and fragmented graphs responsible for poor latency and low GPU occupancy.

## Bounded Goal

Make the public `dots-tts-mf` and `dots-tts-soar` aliases materially faster for ordinary and streaming waveform generation by improving the pure-MLX inference pipeline itself, while preserving speech behavior, model numerics, checkpoint compatibility, and the public API.

## Classification

- Work scale: capability
- Work shape: runtime refactor with behavioral parity and focused performance verification
- Selected lenses: product, engineering, runtime

## Target Stakeholder

Apple Silicon developers using dots.tts locally who care about the time from an inference request to usable audio, including first audio, full waveform completion, and repeated requests on an already loaded model.

## Broader Intent and Scope Preservation

The existing implementation can generate speech, but it remains very inefficient: observed GPU use stays low and end-to-end generation is too slow for the audio duration produced. The user wants inference-time efficiency, not a benchmark suite suitable for publication.

PyTorch and the pinned upstream repositories remain behavioral references for equations, conditioning, cache semantics, solver order, vocoder behavior, and stop rules. They are not the runtime architecture. The implementation must use MLX features and Apple Silicon execution characteristics deliberately rather than copying either the PyTorch control flow or the existing community MLX port.

## Required Outcome

- Replace the uncommitted benchmark-platform work with a small measurement surface: one starting measurement, one final comparison, and focused microtiming only when it decides an implementation choice.
- Audit and optimize the real public inference path, including acoustic autoregression, Qwen/cache/EOS handling, DiT solver execution, AudioVAE bridging, BigVGAN decoding, and streaming state/cadence.
- Remove avoidable host synchronization, repeated context computation, tiny fragmented MLX evaluations, redundant tensor construction, and unstable compilation signatures.
- Use MLX compilation, vectorization, fast primitives, stable-shape execution, explicit request state, and device-resident control where they improve the default path. Use `mx.fast.metal_kernel` only for a proven residual kernel bottleneck that MLX built-ins cannot address cleanly.
- Preserve complete waveform generation. Short output caused by EOS, bounds, streaming flush, or decoder state must be explained by the actual stop semantics and covered by tests; it cannot be labeled a bug or accepted merely because a threshold fired.
- Keep optimizations automatic and private. The public TTS API must not acquire benchmark modes, backend switches, candidate selectors, cache-layout controls, or kernel toggles.

## Constraints and Risks

- Runtime remains pure MLX. Torch, torchaudio, Transformers, `mlx-lm`, `mlx-audio`, native extensions, and upstream packages do not become runtime dependencies.
- Read the pinned upstream PyTorch source before changing each model stage, but never run PyTorch as an inference or performance denominator for this work.
- Public aliases, weights, checkpoint schemas, solver defaults, selective int8 policy, sample rate, and cloning behavior remain unchanged.
- Compiled functions must not capture mutable request RNG, EOS state, cache offsets, recurrent state, or storage shared across concurrent requests.
- Serial model dependencies remain serial. Performance cannot be manufactured by reducing solver work, changing EOS semantics, lowering quality, or batching unrelated requests.
- Stateful vocoder work is alignment-sensitive. Chunk boundaries, lookahead, overlap, final flush, sample count, failure rollback, and interleaved requests require focused parity tests.
- Network/download time is excluded. Local load time is reported separately from request inference.
- Preserve the user's existing `tmp/` contents and all unrelated worktree changes.

## Acceptance Criteria

1. Remove the uncommitted candidate registry, accepted-head ledger, transaction journal, capture orchestration/parser, generalized evidence schema, and related benchmark-platform tests. Retain only diagnostics or tests that directly support inference correctness or a concrete optimization.
2. Use a minimal local measurement command to record the current and final public paths. It reports model load separately and records wall time, time to first audio for streaming, waveform duration, RTF, stop reason, patch count, and peak MLX memory without forcing synchronization inside ordinary inference.
3. Run the starting public-path measurement once. Do not repeat uncached or reference runs unless an implementation failure invalidates that evidence. During development, use focused unit/parity tests and targeted microtiming rather than repeated end-to-end benchmarks.
4. Rework the acoustic loop to eliminate redundant EOS computation/publication, unnecessary host reads, repeated tensor preparation, and unstable compilation while preserving the official rule that the current patch is produced, consumed, and emitted before stopping.
5. Rework DiT execution around reusable MLX graphs or fused operations with stable signatures and explicit request-owned cache state. Growing prefixes, offsets, and request identity must not cause unbounded compilation variants.
6. Rework AudioVAE and BigVGAN so batch execution avoids avoidable eager recurrent work and streaming does not recompute the full decoded left context for every steady chunk. Stateful output must match one-shot output within existing dtype-aware tolerances, including final flush and sample count.
7. Both public aliases produce finite, non-silent 48 kHz waveforms for English and Mandarin, speaker-only and continuation cases. Short, medium, and long completion tests verify target-tail recovery and reject patch-budget exhaustion.
8. Existing deterministic cloning, seeded/interleaved request isolation, failure rollback, checkpoint loading, quality tolerances, and the 30 GiB memory ceiling remain intact.
9. Final default-path measurement is faster than the one starting measurement for both aliases and for batch and streaming. No primary case regresses more than 2%; load and first-request costs remain visible rather than amortized away.
10. Unit tests pass after every code slice. Checkpoint/runtime tiers run only when their owned inference code changes, and local integration runs once for final end-to-end waveform validation.
11. The final implementation contains no benchmark platform, public optimization switch, alternate implementation selector, accepted-head ledger, capture transaction system, or dependency added solely for performance measurement.

## Scope Coverage

### Included

- Pure-MLX implementation work in the dots.tts acoustic model, DiT, cache handling, AudioVAE, BigVGAN, and batch/stream generation paths.
- Focused correctness diagnostics and tests tied to stop behavior, compilation, cache/state publication, decoder alignment, completion, memory, and waveform output.
- Minimal before/after default-path timing and targeted microprofiles used to select an implementation.
- Existing checked-in oracle fixtures and pinned upstream source as read-only behavioral references.

### Deferred / Not in Scope

- Broader quantization, reduced solver steps, approximated math, new checkpoints, or quality/speed tradeoffs.
- Serving infrastructure, multi-request batching, text-arrival streaming, remote publication, and unrelated model families.
- A reusable benchmark, profiling, experiment-management, or trace-processing framework.
- Automated privileged `xctrace`/`powermetrics` collection. A one-off manual trace may be used only when a specific unresolved hotspot requires it, and it remains local scratch evidence.

## Anti-goals

- Do not build a benchmark platform, candidate registry, performance ledger, transaction journal, trace parser, or process supervisor.
- Do not optimize the measurement tools instead of the speech pipeline.
- Do not treat PyTorch or the existing MLX reference implementation as architectural authority.
- Do not rerun uncached/reference pipelines ceremonially or optimize for benchmark repetition.
- Do not claim efficiency from GPU-utilization percentage or a microbenchmark alone; the user-observed inference time and complete waveform are the result.
- Do not improve speed by changing speech math, output length, EOS behavior, weights, precision policy, solver work, or quality thresholds.
- Do not expose internal execution choices through the public API.
