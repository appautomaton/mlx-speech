# Slice 4 Execution Summary

## Status and decision

Complete. DiT request maximum and physical K/V capacity are separate. A default 500-patch request has no physical cache after patch one, allocates bucket 64 on patch two, and grows transactionally through 128, 256, and 512 only as finalized published history requires.

## Changed files

- `src/mlx_speech/models/dots_tts/dit_inference.py`
- `tests/unit/test_dots_tts_dit_cache.py`

## Verification

- Focused cache tests: 29 passed.
- Broader DiT/generation tests: 83 passed.
- Full unit suite: 816 passed.
- MF/SOAR transition tests preserve exact solver output, published K/V values, offsets, dtypes, and branch layout through 64→128→256→512.
- Injected allocation, copy, and materialization failures leave the prior cache unchanged and reusable.
- Scoped Ruff and `git diff --check`: passed.

## Reviews

- Spec compliance: `APPROVED`, no issues.
- Code quality: `APPROVED`, no issues.

## Unresolved risks or next action

Prompt prefill still retains pending projected writes until transactional publication. Slice 5 replaces concatenated prefixes and pending stacks with unpublished scratch while preserving the publication boundary.
