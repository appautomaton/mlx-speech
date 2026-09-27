# Confucius4-R2T2 streaming inference performance

**Status: ACTIVE — sole execution plan for R2T2 streaming performance in
mlx-speech.** Work in small steps: implement, measure, keep what helps.

Closed history: `confucius4-r2t2-streaming.md`. TNT consumer work is parked in
`tnt-asr-live-consumer.md`; do not edit `/Users/ac/dev/ai/tnt-asr`. bf16 only,
no quantization.

| Step | Work | Status |
| --- | --- | --- |
| 1 | Harness and baseline | Done (Entry 1) |
| 2 | Native bf16 streaming compute | Done (Entry 3) |
| 3 | Decoder prefix KV reuse | Done (Entry 3) |
| 4 | Encoder block reuse | Done (Entry 3) |
| 5 | Acceptance run and docs | Pending |

## Target

Apple M5 Max, defaults `chunk_ms=160`, `lookahead_ms=160`,
`unfixed_token_num=1`: p95 whole-step latency ≤ 160 ms and max queue wait
≤ 160 ms (no growing backlog) through 60 s. Report the crossover length for
each build.

## Hard rules

1. Do not change R2T2 behavior: window schedule, prompt, greedy decoding,
   rollback, budget, commit, mel normalization. No skipped windows or
   samples. Context overflow stays an explicit error.
2. Offline `generate` and Qwen3-ASR one-shot keep current defaults. New
   compute is selected per call on the streaming path; no global mode on the
   shared model, no weight copies, no new dependencies.
3. Correctness check for every adopted change: greedy IDs identical to the
   float32-cast reference except numerical ties (reference top-1/top-2
   margin < τ, τ fixed from `test.wav` before gating and never loosened),
   final-transcript CER ≤ 1% relative. Cache reuse (steps 3–4) must match
   the step 2 full-refeed path the same way.
4. Implement first, then test. Compare against the old path only on the
   shortest clip. For longer audio, time only the new path; it passes if it
   is faster than the recorded baseline. Never run both paths at every
   length.
5. `pytest tests/unit/` after each code change (seconds). Checkpoint and
   runtime suites once at the end.

## Step 2 — native bf16 streaming compute

Implemented (per-call `R2T2StreamCompute` / `Qwen3ASRDecodeCompute`):
native-dtype decoder linears and `lm_head`, last-position logits only,
`mx.fast.scaled_dot_product_attention` with GQA, bf16 audio input, bf16
residual. Ablation done (Entry 2): `all` wins. Set it as
`DEFAULT_STREAM_COMPUTE`; amortized PCM storage.

## Step 3 — decoder prefix KV reuse

Keep the session KV cache across windows. Per window, find the first
position whose embedding may differ (start of the open audio block or any
changed audio embedding, then text); truncate KV there at every layer and
prefill the suffix through the offset-aware `decode_step` path (RoPE offset
= retained length). Never reuse KV of generated or rolled-back tokens unless
they are provably identical; the last sampled token may not be in cache.
Amortized cache growth, capped at 65,536.

## Step 4 — encoder block reuse

Diff normalized mel against the previous window; reuse closed 104-token
encoder blocks whose input frames are identical (floor changes and
right-edge reflection invalidate). Recompute the open block.

## Step 5 — acceptance and docs

After steps 2–4 are all in: one precision-gate pass on the real fixtures,
one timing pass at 60 s (p95, max wait). Ask the user for a real ≥60 s
recording if none exists. Write `docs/benchmarks/confucius4-r2t2-streaming-<date>.md` and
create `docs/confucius4-r2t2.md` (usage, streaming compute, measured limits).

## Out of scope

TNT, bounded `feed`/drain contract, quantization, other models, prompt
reordering, VAD, websocket server, offline default changes.

## Evidence log

Fixtures (sha256): `models/netease/confucius4_r2t2/samples/test.wav`
b703174e…7f67 (6.7 s); `outputs/source/hank_hill_ref.wav` 73b90d2c…bc7d
(9.0 s); `peggy_hill_ref.wav` c0b9c21b…038e (13.0 s);
`donald_trump_ref.wav` 9a019dd1…21b3 (11.9 s).

### Entry 0 — 2026-09-26 — reconnaissance (not acceptance)

15 s step ≈ 196 ms: decode steps ~113 ms, prefill ~58 ms, mostly the
per-call float32 weight cast in `_linear_forward`. Native bf16 matmul:
prefill 57.8 → 29.8 ms, decode step 37.7 → 12.3 ms. Cache reuse could save
≤ ~25 ms at 15 s, ~0 under 8 s — hence bf16 compute first.

### Entry 1 — 2026-09-26 — step 1 harness and baseline

- Added `models/confucius4_r2t2/metrics.py`, trace hooks in `streaming.py`,
  adapter stage/work counters, `scripts/eval/benchmark_confucius4_r2t2_streaming.py`,
  unit tests. Queue simulation fixed to FIFO (`start = max(arrival, prev
  finish)`); target requires p95 and max wait ≤ chunk interval.
- Command: `.venv/bin/python scripts/eval/benchmark_confucius4_r2t2_streaming.py
  --chunk-ms 160 --repeats 3 --profile-repeats 1`, reference path.
  Desktop was loaded (WindowServer ~48% CPU); 60 s case stopped as not
  informative for the reference path.

| Audio | p50 | p95 | proc/audio | max wait |
| ---: | ---: | ---: | ---: | ---: |
| 1 s | 214 ms | 299 ms | 1.47 | 511 ms |
| 5 s | 224 ms | 271 ms | 1.37 | 1973 ms |
| 8 s | 202 ms | 252 ms | 1.24 | 2033 ms |
| 15 s | 194 ms | 251 ms | 1.23 | 3539 ms |
| 30 s | 242 ms | 363 ms | 1.57 | 17136 ms |

- Quieter recheck (`--lengths 1 15 --repeats 1`): 1 s p50 164 / p95 176 ms;
  15 s p50 137 / p95 176 ms. Target missed at every length; consistent with
  Entry 0 that decoder weight traffic dominates.
- Unit tests: 1179 passed after the step 2 code landed (last-logits test
  uses a tolerance: Metal fp32 matmul differs between 1-row and N-row
  shapes by ~1e-2).

### Entry 2 — 2026-09-26 — step 2 ablation

`scripts/eval/confucius4_r2t2_compute_ablation.py`, report
`/tmp/r2t2_ablation_1.json`. Preset `all` (native bf16 decoder, last
logits, fused attention, bf16 audio input and residual): p95 58 / 73 /
178 ms at 1 / 15 / 30 s vs reference 170 / 179 / 391 ms; same final text.
Long audio still misses the target because every window recomputes all
history — hence steps 3–4.

### Entry 3 — 2026-09-26 — steps 2–4 implemented

- `DEFAULT_STREAM_COMPUTE` = native bf16 + incremental
  (`asr/_adapters/confucius4_r2t2_incremental.py`): final raw mel frames
  kept, closed 800-frame encoder blocks reused unless the floor change
  affects them, decoder KV kept up to the first changed embedding and only
  the suffix prefilled. Amortized PCM storage; the decoder gets a read-only
  view. KV cache `truncate`/`reserve`; `decode_step(last_logits_only)`.
- Correctness, 4 real clips: incremental vs bf16 full-refeed, 0 generated-ID
  mismatches in 253 windows; final text vs float32 reference, CER 0.
- Timing, new path only, 60 s of concatenated real clips: p50 61 ms, p95
  77 ms, max 112 ms, max queue wait 0 ms. 48–60 s: p95 84 ms. Target met
  through 60 s.
- Unit tests 1179 passed. Checkpoint/runtime suites not yet run.
