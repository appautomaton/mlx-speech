# dots.tts DiT Throughput Optimization Plan

## Goal

Implement the bounded DiT throughput refactor in [SPEC.md](SPEC.md) and retain only exact, production-path improvements.

## Architecture Approach

Keep the existing contiguous scratch cache and fast SDPA as the stable dynamic boundary. Hoist patch-invariant input projection outside NFE loops, then reduce host/kernel fragmentation around attention with explicit-weight compiled functions whose keys depend only on stable tensor geometry. Do not compile mutable request state, cache offsets, or whole solver loops.

## Execution Routing and Topology

Default: direct, serial, and continue through every verified slice. Each candidate is removed immediately when its slice gate fails; no alternate runtime selector remains.

Checkpoints: none.

**Parallel-safe groups:** none

## Ordered Slice Sequence

### Slice 1: Hoist invariant DiT input projections

**Objective:** Project the invariant acoustic tail once per patch and project only changing coordinate tokens inside each NFE.

**Acceptance criteria:**
- MF and SOAR construct the same ten-token DiT sequence values as the existing full projection, bit-for-bit in the production int8/BF16 path.
- SOAR reuses one coordinate projection for conditional and unconditional branches; prompt prefill and first-patch invariant inputs are also reused across NFEs.
- Solver schedules, NFE counts, CFG ordering, masks, positions, cache publication, and public APIs are unchanged.
- One warmed production-shape input-layer A/B is exact and faster; losing code is removed.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_solvers.py
.venv/bin/python -m pytest tests/unit/
git diff --check
```

**Touches:** `src/mlx_speech/models/dots_tts/dit_inference.py`, focused DiT tests

**Produces:** one split input-projection path shared by MF and SOAR

**Status:** complete
**Evidence:** BF16 MF/SOAR now project the invariant acoustic tail once per patch, reuse the SOAR coordinate projection across CFG branches, and reuse prompt-prefill projection across NFEs. A focused production-shape SOAR input-layer check was exact and reduced this subpath from 0.615 ms to 0.418 ms (32.0%); 56 focused DiT/cache/solver tests and 869 unit tests passed, and float32 retains the original full projection to avoid changed matmul rounding.
**Risks / next:** none

### Slice 2: Validate one scratch window per patch

**Objective:** Move invariant cache ownership, capacity, offset, shape, and dtype validation out of every layer/NFE scratch write while preserving transactional publication.

**Acceptance criteria:**
- A private request-scoped scratch window validates the cache once per patch and permits only the expected 18 layers and NFE sequence.
- Growth, publish, close, a different cache/request, or a failed transaction invalidates stale windows.
- First use after allocation/growth performs full validation; `publish_scratch` remains after successful `mx.eval`.
- Existing contiguous storage and attention slices remain unchanged.
- Retain the fast window only if a warmed short cached-tail path improves by at least 1%; otherwise remove the candidate and record the negative result.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_dit.py -k 'scratch or cache or grow or fail or retry or interleav or publish'
.venv/bin/python -m pytest tests/unit/
git diff --check
```

**Depends on:** Slice 1

**Touches:** `src/mlx_speech/models/dots_tts/dit_inference.py`, focused cache tests

**Produces:** a retained validated scratch window or an evidence-backed no-change slice

**Status:** complete
**Evidence:** Added one epoch-bound scratch window per patch; reopening, publish, and cache growth invalidate stale windows, while layer/NFE writes reuse the validated range and delayed offsets still publish only after materialization. One warmed same-loaded production MF cached-tail A/B was exact and improved 18.144 ms to 17.945 ms (1.095%), meeting the retention gate. Focused cache/DiT tests passed 51 cases and the unit suite passed 870 cases.
**Risks / next:** The gain is small and close to the slice threshold; the final request gate decides whether it survives in the combined implementation.

### Slice 3: Compile the stable pre-attention bridge

**Objective:** Reduce the repeated layer-boundary kernels by compiling a stable explicit-weight bridge while leaving dynamic scratch and SDPA eager.

**Acceptance criteria:**
- Prefer a cross-layer bridge combining layer i post-attention tail with layer i+1 norm/modulation/QKV/paired QK normalization/RoPE; use a standalone fixed-ten-token pre-attention island if the bridge fails exactness or speed.
- First-layer preparation and final-layer tail use bounded compiled functions, with callables resolved outside the layer loop.
- Explicit weights preserve BF16 rounding and existing FP32 normalization/RoPE boundaries; modulation weights are not packed.
- Compile key count stays fixed across all 18 layers, MF/SOAR NFEs, offsets, cache growth, and requests.
- The retained boundary is exact and faster after its first-request compile cost is included and amortized over a normal request; failed variants are removed.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_solvers.py -k 'compile or attention or rotary or bf16 or cache or parity'
.venv/bin/python -m pytest tests/unit/
git diff --check
```

**Depends on:** Slice 2

**Touches:** DiT layer/inference code and compiled-boundary tests

**Produces:** one bounded compiled bridge with no runtime selector

**Plan correction:** Both approved boundaries were implemented and tested on the real fused BF16 path. The cross-layer bridge was 9.15% slower over 36 patches and not exact; the standalone pre-attention island was 1.65% slower and not exact. Both changed BF16 fusion/rounding at the QK/RoPE boundary, so retaining either would violate the spec. Slice 3 therefore produces a verified rejection rather than a compiled runtime change; Slice 4 still owns safe host-dispatch reduction around the unchanged eager pre-attention math.

**Status:** complete
**Evidence:** One same-loaded production MF 36-patch pair per approved candidate included first-request compile cost. Cross-layer measured 0.922 s baseline versus 1.006 s candidate; standalone pre-attention measured 0.905 s versus 0.920 s. Both reported non-exact output. All candidate code, tests, caches, and selectors were removed, leaving the verified Slice 2 runtime unchanged.
**Risks / next:** MLX compilation across the BF16 QK/RoPE boundary is excluded unless MLX itself gains exact semantics; do not retry it in Slice 4.

### Slice 4: Remove repeated Python attention dispatch work

**Objective:** Resolve stable runner state once and use a prepared private attention path inside the layer/NFE hot loop.

**Acceptance criteria:**
- Runner initialization validates homogeneous layer/head/norm geometry once.
- Rotary geometry, additive bias, modulation references, and compiled callables are resolved before repeated layer dispatch.
- The private prepared SDPA path skips repeated generic validation but preserves tensor shapes, scale, mask, scratch range, and output exactly.
- No new `.item()` or `mx.eval()` is added inside layer or NFE loops; required patch publication and EOS synchronization remain.
- One baseline and one final steady later-patch trace show lower non-SDPA kernel/CPU-gap overhead with unchanged SDPA call count.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py tests/unit/test_dots_tts_generation.py -k 'attention or cache or compile or interleav or failure'
.venv/bin/python -m pytest tests/unit/
git diff --check
```

**Depends on:** Slice 3

**Touches:** DiT inference runner and attention tests

**Produces:** a prepared hot path with one-time structural validation

**Plan correction:** The prepared SDPA path and once-per-NFE tail-callable resolution were implemented together because both change only repeated host dispatch around the same attention call. Their production A/B was exact but 3.06% slower. The candidate was removed before GPU capture; capturing a rejected path would not affect the retention decision. Existing fast SDPA, prepared bias/rotary geometry, and per-layer validation remain because deleting them did not improve request time.

**Status:** complete
**Evidence:** One same-loaded production MF 36-patch pair measured 0.903 s for the retained generic host path and 0.931 s for the prepared candidate, with array-equal outputs and one stable compiled-tail key. Candidate code and temporary timing script were removed; focused attention/cache/compile/interleaving/failure tests passed 48 cases before removal.
**Risks / next:** Python dispatch remains visible in static structure, but this bounded removal attempt shows it is not an independently profitable target at current MLX costs. Slice 5 decides the combined retained Slice 1-2 result end to end.

### Slice 5: Final production inference gate

**Objective:** Prove the combined retained implementation lowers real inference time without waveform or memory regression.

**Acceptance criteria:**
- Run one warmed same-loaded Hank 261-patch baseline/optimized pair with seed 42; do not repeat uncached trials.
- The waveform is bit-exact, the optimized request is at least 1% faster, and MLX allocator peak grows no more than 2%.
- Report load time separately from inference time; any lazy compile during the first optimized request remains part of inference time.
- MF/SOAR BF16, CFG, cache, failure/retry, and public generation tests pass.
- Remove temporary timing/trace scripts after use; keep any requested audio only under `/private/tmp`.

**Verification:**
```bash
.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/
.venv/bin/ruff check src/mlx_speech/models/dots_tts src/mlx_speech/generation/dots_tts.py tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_dit_cache.py
git diff --check
```

**Depends on:** Slice 4

**Touches:** retained DiT runtime/tests and temporary `/private/tmp` evidence only

**Produces:** exact end-to-end timing and memory evidence

## Aggregate Verification Commands

| Gate | Command | When |
| --- | --- | --- |
| Focused DiT/cache | slice-local commands above | every slice |
| Required default | `.venv/bin/python -m pytest tests/unit/` | every retained slice |
| Inference tiers | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/` | final retained implementation |
| Final real request | one same-loaded 261-patch baseline/optimized pair | Slice 5 only |
