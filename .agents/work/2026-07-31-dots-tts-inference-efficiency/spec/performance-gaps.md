# Normative Performance Gap Map

This file is part of the [default-path inference efficiency specification](../SPEC.md). Each gap must be closed by behavior-preserving implementation evidence; timing alone cannot waive an invariant.

| Gap | Starting evidence | Required closure | Verification signal |
| --- | --- | --- | --- |
| GAP-DEC-01 | Non-streaming generation drains 1/1/4 fixed-window decode, while pinned upstream batch-decodes once. | Shared latent generation feeds a batch sink for ordinary generation and a bounded streaming sink for `generate_stream()`. | Batch/stream patch and sample counts match; waveform and seam tolerances pass; total decode time is reported separately. |
| GAP-DEC-02 | BF16 decoder weights receive FP32 latent/state arrays, risking FP32 promotion through BigVGAN. | BigVGAN entry and rolling decoder window use decoder weight dtype; any FP32 SLSTM boundary is justified by evidence. | Real checkpoint dtype assertions, decoder timing, waveform/seam parity, and quality gate. |
| GAP-DEC-03 | Four SLSTM layers are Python-unrolled per frame and every chunk rebuilds decoder work. | Common chunk shapes reuse compiled/fused recurrence and decoder execution without sharing request state. | Compile-cache hit coverage, first/warm timing, state parity, early-close safety. |
| GAP-DIT-01 | A 500-patch request selects a 512 bucket before actual output length is known. | Allocate from current finalized history and grow through official buckets with transactional prefix copy. | Patch-two allocation evidence, transition tests, two-patch maximum-bucket memory smoke. |
| GAP-DIT-02 | Each layer/NFE concatenates the complete cached K/V prefix with fresh tail tensors. | Attend over one contiguous published-prefix-plus-fresh-tail view without copying full history or prematurely publishing the noisy tail. | Oracle parity, allocation-by-patch profile, failure rollback, exact offsets/content. |
| GAP-DIT-03 | QKV and adaptive modulation remain separate checkpoint-shaped projections and hot steps are not compiled. | Build inference-only packed projections and reuse stable compiled/fused first, prefill, and later steps when timing proves value. | Packed/unpacked numerical parity, no serialization changes, first/warm stage timing. |
| GAP-PROMPT-01 | Repeated reference audio reruns speaker and optional AudioVAE prompt encoders. | Bounded content-aware cache stores reusable speaker features and pre-sampling latent distributions. | Encoder call counts, invalidation cases, cold/warm TTFC, deterministic seeds/scales. |
| GAP-SYNC-01 | Patch, cache, decoder state, waveform, finite, and silence checks create redundant host/device boundaries. | Keep only boundaries required for control flow, transactional publication, and safe yielding; combine reductions. | Instrumented eval/item counts plus injected failures and early iterator close. |
| GAP-AUX-01 | MeanFlow builds unused CFG history; Qwen/semantic fixed decode shapes rebuild avoidable work. | Remove mode-dead state first; optimize Qwen/semantic only if residual stage timing justifies it. | Retained-array counts, MF/SOAR parity, stage profile before and after. |

## Evidence Rules

- The frozen comparison is cached starting path versus cached completed path on the same host and artifacts.
- Exactly matched inputs, seeds, solver settings, patch counts, and output-health requirements are mandatory.
- Uncached execution is a correctness oracle only. It is not a repeated timing workload or a release-claim denominator.
- A speed improvement does not close a gap if deterministic output, quality, bounded memory, request isolation, or failure atomicity regresses.
- Existing uncommitted compact-tail and benchmark files are starting-tree material, not accepted evidence. Planning must explicitly adopt, revise, or remove them before the baseline is frozen.
