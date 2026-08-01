# Slice 6 Execution Summary

## Status and decision

Complete as a rejected experiment. No Slice 6 runtime or test change is retained. End-to-end cached DiT-stage evidence showed that both full QKV/AdaLN packing and selective AdaLN packing made the default path slower, so the committed Slice 5 implementation remains authoritative.

## Changed files

- No production or test files.
- This evidence summary and the PLAN status are the only durable changes.

## Verification and decision evidence

- Exact plan test command on the restored Slice 5 tree: 64 passed, including available MF/SOAR base and int8 checkpoint loads.
- QKV component packing was removed because its BF16 maximum difference was `0.015625`; the full packed profile changed MF waveform peak from `0.7265625` to `0.71875`.
- Full QKV/AdaLN packed profile versus Slice 5: MF acoustic `1.205888s`→`1.484992s` (+23.1%) and peak +344,168,192 bytes; SOAR acoustic `4.043478s`→`4.537410s` (+12.2%) with a similarly material process-memory increase.
- Selective AdaLN used a single packed backing buffer with serialized parameter names rebound to views. It was bit-exact, added only 4,096 active bytes, and improved isolated batch-1/batch-2 modulation microkernels, but the representative 32-patch profile still regressed MF acoustic time to `1.474046s` (+22.2%) and SOAR to `4.567270s` (+13.0%).
- Selective diagnostic SHA-256: `60f7ff770f967d0d4d80b334f8b7184e8d13e43e4ba9b11d9a791a217d939757`.
- Scoped Ruff and `git diff --check`: passed; the complete Slice 6 production/test diff is empty.

## Reviews

No implementation review was required because no runtime or test code survived the performance gate. The coordinator rejected the experiment under the SPEC's cached default-path efficiency objective; inference packing is optional in the SPEC and may not displace a faster default implementation.

## Unresolved risks or next action

None. Do not revisit QKV/AdaLN packing during this change unless MLX kernel behavior changes or a new design avoids both fused-shape slowdown and duplicate backing storage.
