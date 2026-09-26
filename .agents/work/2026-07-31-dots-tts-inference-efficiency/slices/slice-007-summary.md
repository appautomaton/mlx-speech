# Slice 7 Execution Summary

## Status and decision

Complete. Each generator owns a lock-protected 256-entry memory-only LRU keyed by normalized reference content and prompt mode. Entries contain only materialized unscaled speaker embeddings and, for continuation, pre-sampling AudioVAE distributions. Speaker scale and request seed remain outside cached values.

## Changed files

- `src/mlx_speech/generation/dots_tts.py`
- `tests/unit/test_dots_tts_generation.py`

## Verification

- Focused prompt/reference/speaker/cache/seed/interleaving selection: 28 passed, 36 deselected.
- Full unit suite: 824 passed.
- Same-content arrays and paths reuse eligible encoders; changed normalized content and different prompt modes miss.
- Different scales reproject the cached unscaled embedding; different seeds resample the cached distribution; repeated seeds reproduce prompt latents.
- The LRU evicts deterministically at 256 entries and exposes explicit clearing for profiler/model lifetime control.
- Concurrent same-key misses compute outside the cache lock, converge on one canonical entry, and do not retain duplicate entries.
- Scoped Ruff, format check, and `git diff --check`: passed.

## Reviews

Direct-route coordinator review found no unresolved correctness or lifecycle issue. Cache contents are materialized before insertion, request-local sampled latents are never cached, and all mutation is serialized under the dedicated lock.

## Unresolved risks or next action

The final cached profiler measures cold-reference request one and same-reference warm requests two and three. No additional prompt microbenchmark is needed before that canonical comparison.
