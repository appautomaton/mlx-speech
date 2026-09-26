# Slice 5 Execution Summary

## Status and decision

Complete. DiT attention now writes two fresh units into unpublished layer-local scratch, attends over one contiguous published-prefix-plus-scratch view, and publishes exactly one finalized five-token unit only after every NFE and cache array materializes. The initial unified six-dimensional update design was rejected after it regressed MF/SOAR; the accepted per-NFE/per-layer storage removes that execution pathology.

## Changed files

- `src/mlx_speech/generation/dots_tts.py`
- `src/mlx_speech/models/dots_tts/dit_inference.py`
- `tests/unit/test_dots_tts_dit_cache.py`
- `tests/unit/test_dots_tts_generation.py`

## Verification

- Focused Slice 5 tests: 40 passed, 40 deselected.
- Full unit suite: 820 passed.
- Exact MF/SOAR trusted-solver parity covers CFG, BF16, continuation, progressive growth, cached interleaving, and non-aliased request storage.
- Injected mid-NFE failure tests snapshot identical published prefixes and offsets before failure, after failure, and after retry.
- Cached 32-patch Slice 4→5 A/B: MF acoustic `1.281609s`→`1.205888s` (-5.91%) and total `7.472466s`→`7.459050s`; SOAR acoustic `4.350379s`→`4.043478s` (-7.05%) and total `10.716186s`→`10.338544s`. SOAR peak decreases by 52,664 bytes; the MF 28,288-byte delta is 0.00030% process-level measurement noise and reverses during warmup.
- Slice 4 reference diagnostic SHA-256: `14b2e051183a9e89f4e45c7f8d5915353762daf2ce7aa1961b460ac949d7090f`.
- Slice 5 diagnostic SHA-256: `9085188a9e17182485409a5633ab05f16257f2d7eaea751d3a1243e77fed25c6`.
- Scoped Ruff and `git diff --check`: passed.

## Reviews

- Spec compliance: `APPROVED`, no issues after adding direct cached MF/SOAR interleaved-request isolation.
- Code quality: `APPROVED`, no issues.

## Unresolved risks or next action

The supplied production-checkpoint profile is bounded to 32 patches. Final 128-patch comparison in Slice 10 verifies long-duration behavior after all optimizations.
