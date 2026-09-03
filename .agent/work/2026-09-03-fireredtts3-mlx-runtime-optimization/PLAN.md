# FireRedTTS3 MLX Runtime Optimization Plan

## Goal

Implement the approved [runtime-first optimization specification](SPEC.md): close RedAE sliding-attention parity and retain only benchmark-proven MLX runtime improvements without changing the artifact or public API.

## Architecture approach

Keep the shared Qwen3 full-attention and KV-cache paths behaviorally unchanged. Add a distinct window-bounded causal path only for configurations that request sliding attention, then prove it numerically against the current dense-mask equation. RedAE wires its encoder and decoder configs explicitly; its three-token CLS downsampler remains full attention.

Treat compilation, fused attention, and cached rotary/time constants as measured candidates rather than mandatory architecture. A repository audit command owns repeatable stage timing, MLX memory capture, golden inputs, and pass/fail thresholds. Slice 1 records a parity-correct baseline before hot-path work begins; Slice 2 compares every candidate with that baseline and the pre-parity profile from the specification. Any candidate that misses the gates is removed before the slice completes.

Compiled functions must be pure tensor-in/tensor-out regions with stable shapes. KV-cache append, cache-length mutation, stop-score `.item()` synchronization, and request lifecycle remain outside compiled regions. Attention length comparisons run in independent subprocesses and use peak deltas above each subprocess's initialized baseline so allocator history cannot create a false memory result.

## Execution routing and topology

Execution is direct and serial: Slice 1 → Slice 2 → Slice 3. Continue through every approved slice after verification passes.

**Parallel-safe groups:** none.

No slice has a human checkpoint. Slice 2's benchmark decides which optimization candidates remain; it does not reopen product scope.

## Ordered slice sequence

### Slice 1: Correct and bound RedAE sliding attention

**Objective:** Match the official RedAE attention configuration and replace dense masked sliding attention with a numerically equivalent window-bounded path.

**Acceptance criteria:**

- Encoder layers read `enc_sliding_window=64`; decoder layers read `dec_sliding_window=64`; CLS downsampling uses full attention.
- Deterministic attention fixtures match the dense reference before, across, and after the 64-token boundary within declared BF16/FP32 tolerances.
- Full-attention and autoregressive KV-cache paths remain unchanged for Qwen3-ASR and FireRed's Base core.
- A 512-versus-2048-token diagnostic runs each length in an independent subprocess, subtracts its initialized MLX baseline, shows peak-delta growth no greater than 6× for the 4× sequence increase, and returns finite values.
- The existing RedAE artifact strictly loads and executes a finite encode/decode pass.
- Before Slice 2 changes the hot path, three warm golden runs record the parity-correct stage medians and MLX peak in `/tmp/fireredtts3-mlx-port/parity-baseline.json`; cold load and one unmeasured warmup are excluded.

**Touches:** `src/mlx_speech/models/fireredtts3/redae.py`, `src/mlx_speech/models/qwen3_asr/text_decoder.py`, `tests/unit/test_fireredtts3_redae.py`, `tests/unit/test_qwen3_asr_text_decoder.py`, `tests/checkpoint/test_fireredtts3_audio_checkpoint.py`, `scripts/audit/fireredtts3_runtime.py`

**Produces:** parity-correct, window-bounded RedAE attention plus a reusable local runtime audit command.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_fireredtts3_redae.py tests/unit/test_qwen3_asr_text_decoder.py tests/unit/test_qwen3_asr_generation.py
MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_audio_checkpoint.py
.venv/bin/python scripts/audit/fireredtts3_runtime.py attention --lengths 512 2048 --window 64 --isolated --max-peak-growth 6.0
.venv/bin/python scripts/audit/fireredtts3_runtime.py profile --model-dir models/firered/firered_tts3/mlx-bf16 --reference-audio /tmp/fireredtts3-mlx-port/reference.wav --reference-text "For Timothy was a spoiled cat, and he allowed no one." --text "你好，很高兴认识你。" --language Chinese --seed 1234 --flow-steps 10 --guidance-scale 2.0 --warmup-runs 1 --runs 3 --output /tmp/fireredtts3-mlx-port/parity-baseline.json
```

**Status:** complete
**Evidence:** corrected encoder/decoder/CLS window wiring and added a fused GQA
sliding path using 64-query blocks with local causal masks. Twenty focused
RedAE/shared-Qwen tests and the strict real audio-checkpoint test passed. In
isolated processes, attention peak delta grew 4.23× for a 4× sequence increase
(512→2048), with finite output. The saved three-run parity baseline records
17 patches, cache 48→65, median core 2.6478 seconds, and median MLX peak
6,193,104,023 bytes.
**Risks / next:** the first isolated timing included Metal warmup, so Slice 2
uses the stable three-run golden profile rather than attention micro-timing.

### Slice 2: Retain benchmark-proven MLX hot-path improvements

**Objective:** Evaluate and retain only MLX-native changes that materially reduce the measured Qwen3/DiT generation latency or MLX peak memory.

**Acceptance criteria:**

- The audit command reports cold load separately and records one excluded warmup followed by at least three measured warm runs with stage medians, generated patch count, cache length, MLX active/cache/peak memory, and process peak.
- Fixed-shape DiT/PatchEncoder, Qwen decode, fused attention, and rotary/timestep constant caching are evaluated where applicable; rejected candidates leave no production code behind.
- A retained candidate improves its selected metric by at least 5% over the post-Slice-1 parity baseline, and the difference between medians exceeds the larger of the baseline and candidate three-run ranges; the final implementation also improves median core-generation latency or MLX peak by at least 10% against the pre-parity 2.556 seconds / 5.770 GiB profile, while the other metric regresses by no more than 10% against either baseline.
- Qwen performs one prompt prefill followed by one-token continuation, uses grouped KV heads, rejects cache overflow, and preserves deterministic seed behavior.
- After one unmeasured warmup, three consecutive requests on one loaded model each release their result, run Python collection, and clear the MLX allocator cache; the maximum minus minimum post-cleanup active memory is no greater than 64 MiB.
- No compiled function mutates `Qwen3ASRTextKVCache`, reads its Python `current_length`, performs stop-score `.item()` synchronization, or owns request cleanup.
- Stop logic, two-patch history, cosine schedule, Euler update, CFG, PatchEncoder, and DiT numerical equations remain covered by focused tests.

**Depends on:** Slice 1

**Touches:** `src/mlx_speech/models/fireredtts3/core.py`, `src/mlx_speech/models/fireredtts3/dit.py`, `src/mlx_speech/models/fireredtts3/patch_encoder.py`, shared Qwen3 primitives only when the candidate benefits and preserves both families, `scripts/audit/fireredtts3_runtime.py`, `tests/unit/test_fireredtts3_core.py`, `tests/unit/test_fireredtts3_generation.py`, `tests/unit/test_qwen3_asr_text_decoder.py`, `tests/runtime/test_fireredtts3_runtime.py`

**Produces:** a measured optimized runtime and repeatable before/after performance evidence.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_fireredtts3_core.py tests/unit/test_fireredtts3_generation.py tests/unit/test_qwen3_asr_text_decoder.py tests/unit/test_qwen3_asr_generation.py
MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/runtime/test_fireredtts3_runtime.py
.venv/bin/python scripts/audit/fireredtts3_runtime.py profile --model-dir models/firered/firered_tts3/mlx-bf16 --reference-audio /tmp/fireredtts3-mlx-port/reference.wav --reference-text "For Timothy was a spoiled cat, and he allowed no one." --text "你好，很高兴认识你。" --language Chinese --seed 1234 --flow-steps 10 --guidance-scale 2.0 --warmup-runs 1 --runs 3 --parity-baseline /tmp/fireredtts3-mlx-port/parity-baseline.json --baseline-core-seconds 2.556 --baseline-mlx-peak-gib 5.770 --min-candidate-improvement 0.05 --min-total-improvement 0.10 --max-other-regression 0.10
```

**Status:** complete
**Evidence:** retained FireRed-local BF16 linear execution and compilation of
only the fixed-shape DiT tensor region. The compiled and eager DiT outputs match
within `atol=1e-6`, `rtol=1e-5`; mutable KV cache, stop synchronization, and
request cleanup remain eager. The formal three-run profile passed its gate at a
2.0518-second median core time: 22.5% faster than the parity-correct baseline
and 19.7% faster than the historical baseline, with a 6,193,075,587-byte median
MLX peak (effectively unchanged). Twenty-one focused unit tests and the strict
three-request runtime cleanup test passed.
**Rejected candidates:** fused full attention improved core time only 2.3% and
raised peak memory about 2%; compiled single-patch PatchEncoder regressed the
retained candidate by about 2.9%. Qwen compilation was inapplicable because its
continuation path owns variable cache tensors and Python cache length. Stable
DiT rotary/timestep construction is already inside the retained compiled graph;
shape-growing Qwen rotary caching was not retained. No rejected candidate leaves
production code behind.
**Risks / next:** local BF16 compute and DiT compilation can alter floating-point
rounding while staying equation-compatible, so Slice 3 must re-run bitwise seeded
waveform, ASR, and speaker-similarity gates before completion.

### Slice 3: Revalidate public waveform quality and compatibility

**Objective:** Prove that the optimized runtime preserves the existing artifact, API, deterministic generation, and golden voice-cloning result.

**Acceptance criteria:**

- The existing flat artifact loads without conversion or tensor-name changes, and the unified API/CLI signatures remain unchanged.
- Two identical seeded requests produce bitwise-equal waveform output on the same loaded model; repeated-request post-cleanup active-memory spread remains at or below 64 MiB after the first warmup.
- The golden output is finite, non-silent, mono 24 kHz; local ASR returns `你好，很高兴认识你。` exactly and CAM++ reference/output cosine is at least 0.70.
- FireRed checkpoint, runtime-purity, shared Qwen3-ASR, complete unit, runtime, and local integration suites pass with no required checkpoint skips.
- The FireRed guide records the measured optimization result and retains the Base-only/runtime-only limitations.

**Depends on:** Slice 2

**Touches:** `tests/runtime/test_fireredtts3_runtime.py`, `tests/integration/test_fireredtts3.py`, `tests/unit/test_fireredtts3_dependency_guard.py`, `docs/fireredtts3.md`

**Produces:** final compatibility, quality, memory, and performance evidence for the optimized runtime.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/
MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_audio_checkpoint.py tests/checkpoint/test_fireredtts3_core_checkpoint.py
MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/runtime/test_fireredtts3_runtime.py
RUN_LOCAL_INTEGRATION=1 MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/integration/test_fireredtts3.py
```

**Status:** complete
**Evidence:** the unchanged flat BF16 artifact and public TTS API generate mono
24 kHz finite, non-silent audio. Two requests on one loaded model with the
golden arguments produced bitwise-identical waveforms. Local Qwen3-ASR returned
`你好，很高兴认识你。` exactly, and reference/output CAM++ cosine was
0.7374283671. The strengthened end-to-end three-request cleanup fixture stayed
within the 64 MiB active-memory bound. The full 1,103-test unit suite, all three
FireRed strict checkpoint tests, the runtime test, and the local integration
test passed with no required-checkpoint skips. The guide now records the
measured runtime result, compiled/eager boundary, quality evidence, and existing
Base-only limitations.
**Risks / next:** none within the approved optimization scope; benchmark values
remain local-machine comparisons rather than portable performance guarantees.

## Aggregate verification commands

| Scope | Command |
| --- | --- |
| Shared fast regression | `.venv/bin/python -m pytest tests/unit/` |
| FireRed real checkpoints | `MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_audio_checkpoint.py tests/checkpoint/test_fireredtts3_core_checkpoint.py` |
| Repeated-request runtime | `MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/runtime/test_fireredtts3_runtime.py` |
| Golden waveform | `RUN_LOCAL_INTEGRATION=1 MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/integration/test_fireredtts3.py` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan separates parity attribution from candidate gains, isolates allocator measurements, quantifies steady-state memory, and keeps mutable request state outside compilation.
- Concern: Slice 1 can satisfy the memory gate with a per-token Python window loop that materially regresses long-sequence latency, so execution must use bounded query blocks and report diagnostic timing.
- Concern: Slice 2 can trigger shape-specific recompilation if variable-length KV tensors enter a compiled function, so only stable tensor regions should remain compiled after the warm-run evidence.
- Action: Execute serially, reject any window or compile candidate that violates the repeated timing ranges, and preserve the uncompiled path until its replacement passes all gates.
- Verified: Prior corrections confirmed; RedAE and core data flow, independent-process peak deltas, post-parity attribution, 64 MiB cleanup tolerance, mutable cache boundaries, rollback, and verification commands reviewed.
