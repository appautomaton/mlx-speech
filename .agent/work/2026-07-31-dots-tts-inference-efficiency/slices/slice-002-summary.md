# Slice 2 Execution Summary

## Status and decision

Complete. Ordinary synthesis now collects payload latents from the shared request-local producer and calls `AudioVAE.decode()` once. Streaming alone owns bounded 1/1/4 recurrent decoding, flushing, and early-close cleanup.

## Changed files

- `src/mlx_speech/generation/dots_tts.py`
- `src/mlx_speech/tts/_adapters/dots_tts.py`
- `tests/unit/test_dots_tts_generation.py`
- `tests/unit/test_dots_tts_adapter.py`

## Verification

- Focused Slice 2 tests: 56 passed.
- Full unit suite: 798 passed.
- Scoped Ruff and `git diff --check`: passed.

## Reviews

- Spec compliance: `APPROVED`, no issues.
- Code quality: `APPROVED`, no issues.

## Unresolved risks or next action

Checkpoint-backed behavior was not exercised in this unit-only slice. Slice 3 owns the real-checkpoint decoder precision, waveform, seam, and focused quality gates.
