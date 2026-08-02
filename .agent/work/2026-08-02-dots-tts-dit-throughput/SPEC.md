# dots.tts DiT Throughput Optimization Spec

**Bet:** Hoisting invariant DiT work and reducing host dispatch around the existing MLX kernels will lower full-request inference time without changing generated audio or increasing memory materially.

## Goal

Make the pure-MLX dots.tts MF and SOAR inference paths faster by addressing repeated DiT projection work, scratch-cache host overhead, fragmented pre/post-attention kernels, and per-layer Python dispatch while preserving the current solver, quantization, cache, and waveform behavior.

**Work scale:** capability

**Work shape:** performance refactor

**Selected lenses:** product, engineering, runtime

## Required Outcome

- Project the invariant six-token acoustic tail once per patch and project only the four changing coordinate tokens per NFE; reuse SOAR coordinate projections across CFG branches and reuse prompt-prefill projections across NFEs.
- Keep the proven contiguous unpublished K/V scratch layout, but remove repeated Python validation from each layer/NFE write when a request-scoped validated window can do so safely.
- Reduce short-kernel and Python scheduling overhead with a bounded compiled pre-attention boundary or cross-layer bridge. Dynamic scratch writes and SDPA remain outside compilation.
- Validate homogeneous DiT layer geometry once and use a private prepared attention path without repeated public-API checks inside the hot loop.
- Keep all state publication after successful MLX materialization and preserve failure, retry, growth, close, and interleaved-request semantics.

## Constraints and Risks

- Pure MLX runtime; PyTorch is an equation and behavior reference only.
- Qwen eligible weights remain int8 with BF16 activations/scales/bias. DiT and semantic execution remain BF16 with existing intentional FP32 normalization, RoPE, and solver boundaries.
- Do not change MF/SOAR schedules, NFE counts, CFG, EOS, noise, masks, positions, audio length, or public APIs.
- Compile keys must not include logical prefix length, request offset, layer identity, NFE index, or request state. Compilation must amortize within a normal 35-36-patch request.
- Retain an optimization only when it is waveform-exact and improves its targeted warmed production-shape path. The final combined path must improve one same-loaded 261-patch Hank request by at least 1%, with MLX allocator peak growth no greater than 2%.
- MLX allocator peak is not total process RSS and must be reported as such.
- Cache-window validation elision is expected to be small and is retained only if its targeted short hot path improves by at least 1%.

## Acceptance Criteria

- MF and SOAR first-patch, later-patch, CFG, prompt-prefill, cache growth, and BF16 fixtures remain exact.
- Cache failure/retry, stale-window rejection, close, publication ordering, and interleaved requests remain correct.
- The retained compile path has fixed key cardinality across layers, NFEs, offsets, cache growth, and later requests.
- A single before/after steady-patch GPU trace shows reduced non-SDPA kernel/CPU-gap overhead without changing MF 72 or SOAR 180 SDPA calls per patch.
- One warmed same-loaded 261-patch Hank baseline/optimized pair is waveform-exact, at least 1% faster end to end, and has no more than 2% MLX allocator peak regression.
- `pytest tests/unit/` passes; because inference code changes, focused checkpoint/runtime coverage also passes once after the retained implementation is complete.

## Anti-goals

- No benchmark platform, experiment registry, repeated uncached trials, or runtime implementation selector.
- No broader quantization, forced int8 conversion, solver approximation, NFE reduction, EOS relaxation, or audio truncation.
- No retry of previously rejected full DiT compilation, QKV/AdaLN packing, fast RoPE, unified 6D cache updates, or vocoder tiling variants.
- GPU utilization percentage is diagnostic evidence, not an optimization target by itself.
