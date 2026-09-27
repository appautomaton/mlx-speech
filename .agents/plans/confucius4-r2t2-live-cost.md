# Confucius4-R2T2 streaming inference performance

**Status: DONE / CLOSED (2026-09-26).** The user confirmed live use in TNT
is much faster. Do not append tasks.

Closed history: `confucius4-r2t2-streaming.md`. TNT consumer work is parked in
`tnt-asr-live-consumer.md`; do not edit `/Users/ac/dev/ai/tnt-asr`. bf16 only,
no quantization.

| Step | Work | Status |
| --- | --- | --- |
| 1 | Harness and baseline | Done (Entry 1) |
| 2 | Native bf16 streaming compute | Done (Entry 3) |
| 3 | Decoder prefix KV reuse | Done (Entry 3) |
| 4 | Encoder block reuse | Done (Entry 3) |
| 5 | Acceptance | Done |

## Target

Streaming keeps up with live speech at the defaults (`chunk_ms=160`,
`lookahead_ms=160`, `unfixed_token_num=1`). The user judges speed in real use;
agent timings are not reliable (power mode varies) and are not recorded.

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
   shortest clip. Speed is judged by the user, not by agent timings.
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

## Step 5 — acceptance

Checkpoint/runtime regression suites once. No long recording and no docs
(user decisions 2026-09-26); live use in TNT was confirmed faster by the
user.

## Out of scope

TNT, bounded `feed`/drain contract, quantization, other models, prompt
reordering, VAD, websocket server, offline default changes.

## Evidence log

Fixtures (sha256): `models/netease/confucius4_r2t2/samples/test.wav`
b703174e…7f67 (6.7 s); `outputs/source/hank_hill_ref.wav` 73b90d2c…bc7d
(9.0 s); `peggy_hill_ref.wav` c0b9c21b…038e (13.0 s);
`donald_trump_ref.wav` 9a019dd1…21b3 (11.9 s).

### Entry 0 — reconnaissance

The per-call float32 weight cast in `_linear_forward` dominated each step;
native bf16 matmuls remove it. Cache reuse saves little in the first 8 s.

### Entry 1 — step 1 harness

`models/confucius4_r2t2/metrics.py`, trace hooks, adapter counters,
`scripts/eval/benchmark_confucius4_r2t2_streaming.py`, FIFO queue simulation,
unit tests.

### Entry 2 — step 2 compute

`scripts/eval/confucius4_r2t2_compute_ablation.py`: preset `all` (native
bf16 decoder, last-position logits, fused attention, bf16 audio input and
residual) adopted; same final text as the reference.

### Entry 3 — steps 2–4 implemented

- `DEFAULT_STREAM_COMPUTE` = native bf16 + incremental
  (`asr/_adapters/confucius4_r2t2_incremental.py`): final mel frames kept,
  closed 800-frame encoder blocks reused unless a floor change affects them,
  decoder KV kept up to the first changed embedding, suffix-only prefill.
  Amortized PCM storage; KV `truncate`/`reserve`.
- Correctness on 4 real clips: 0 generated-ID mismatches vs the bf16
  full-refeed path; final text identical to the float32 reference.
- Qwen3-ASR offline output identical before/after on the same clips; unit,
  R2T2, and FireRedTTS3 runtime tests pass.
- User confirmed live use in TNT is much faster.
