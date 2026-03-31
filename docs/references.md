# Upstream References

This repository uses `.references/` for optional local checkouts of upstream
projects that are useful for source inspection.

These checkouts are not part of the `mlx-speech` runtime, build, or packaging
story. They exist so implementation work can refer to upstream code locally
without turning those projects into vendored dependencies.

## Current Policy

- Keep upstream references shallow when possible.
- Prefer released code when a stable public release exists.
- Prefer current default-branch heads for fast-moving reference projects when we
  are studying implementation patterns rather than pinning a runtime dependency.

## Planned References

As of March 29, 2026:

- `mlx`: latest public GitHub release `v0.31.1`
- `MOSS-TTS`: shallow clone of `main`
- `MOSS-TTSD`: shallow clone of `main`
- `mlx-audio`: shallow clone of `main`

## Current Checkouts

- `.references/mlx`: `v0.31.1` at `ce45c52`
- `.references/MOSS-TTS`: `main` at `c74844ef6c08161160483c1bf3682235bdccae41`
- `.references/MOSS-TTSD`: `main` at `20dbb4fc44819435fee894d644a0402a0fee736a`
- `.references/mlx-audio`: `main` at `6408d2a410eb8c57464e07725b92271860199250`
- `.references/transformers`: `main` at `8213e0d920d52cb00dcade16b6d1f6e952ac0a8c` (sparse: `src/transformers/models/cohere_asr`, `src/transformers/models/moonshine`, `src/transformers/models/parakeet`)

## Notes

- `mlx` is a real dependency of the project, but the checkout in
  `.references/mlx` is for local source inspection only.
- `MOSS-TTS`, `MOSS-TTSD`, and `mlx-audio` are reference codebases, not runtime
  dependencies.
- `MOSS-TTS` appears to be the active family repository and is the best primary
  OpenMOSS reference point going forward.
