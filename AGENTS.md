# mlx-voice

Always address the user as **My Love** at the beginning of your responses.

## GPT / Codex Agents

> If you are a GPT-based model or Codex, also read `CODEX.md` for
> response-quality rules that apply to you.

## Current Plan

> **Active:** `plans/v1-moss-tts.active.md`
> **Done:** `plans/v2-vibevoice.done.md`
>
> Read the active plan before starting implementation work.

## Mission

`mlx-voice` is an open-source, MLX-native speech library for Apple Silicon.
The long-term objective is a clean library that supports multiple speech model
families behind a consistent interface without becoming a dependency-heavy
umbrella framework.

## Non-Negotiables

- Runtime must be pure MLX.
- Checkpoint handling and remapping must also stay in MLX.
- Do not ship torch-backed inference or torch-based conversion behind an MLX
  label.
- Upstream PyTorch repos are references only, not the runtime or conversion
  design center.
- Keep the public library surface clean enough for long-term OSS maintenance.
- End-to-end means waveform output. A token-only path is not complete speech
  inference.

## Dependency Stance

Keep dependencies minimal by default. Add packages only when the
implementation proves they are necessary.

- `mlx`: yes
- `numpy`: yes
- `safetensors`: yes
- `torch`: no
- `torchaudio`: no
- `huggingface_hub`: avoid until the Python library itself truly needs it
- `mlx-audio`: reference project, not a required dependency
- Machine-global tools (`hf` CLI): not project dependencies

## Architecture Principles

- Separate runtime inference from checkpoint conversion.
- Design around model adapters, not around one upstream repo's layout.
- Prefer local-path-first loading and explicit weight remapping.
- Use `.safetensors` as the preferred checkpoint format.
- Avoid PyTorch-shaped abstractions in the MLX runtime.
- Keep model code in the package (`src/`) and model weights outside (`models/`).
- Weights are never committed to git. Future publication goes to HuggingFace.

## Repository Shape

```
src/mlx_voice/      # Published library code
models/             # Local checkpoints, not packaged, not in git
plans/              # Versioned implementation plans
tests/              # Focused package tests
examples/           # Small usage examples
scripts/            # Conversion and maintenance helpers
docs/               # Internal project notes
.references/        # Read-only upstream checkouts for source inspection
```

## Upstream References

- `.references/` checkouts are for reading, comparison, and mapping logic.
- Do not treat them as vendored runtime code.
- Keep them shallow when possible.
- Document pinned commits in `docs/references.md`.
- **Read upstream source before implementing.** Do not guess architecture from
  names or documentation alone.

## Working Rules

- Work from `/Users/ac/dev/ai/genai/mlx-voice`.
- Prefer clarity and optionality while the architecture is forming.
- Surface design choices when they affect long-term API, packaging, or
  dependency weight.
- Keep comments and docs short, explicit, and high-signal.
- Scope is defined in the active plan. Do not broaden beyond it.
- Tackle unfinished work iteratively. Finish one clear slice, validate it,
  update the active plan, then move to the next slice.
- Keep runtime, conversion, tests, and helper scripts organized in their
  existing boundaries. Do not blur file responsibilities.

## Runtime State

The `MossTTSLocal` v0 runtime is complete and operational. Do not treat it as
a skeleton or a bring-up project.

- Default runtime: `mlx-int8` quantized weights, `W8Abf16` mixed precision,
  global + local KV cache enabled for single-item sampled inference.
- The uncached path (`--no-kv-cache`) exists as a debug and comparison
  fallback. It is not the primary path.
- KV cache default is a settled decision. Do not re-open it.
- Implemented inference modes: direct generation, clone, continuation,
  continuation + clone.

## Validation

- Prefer small, direct checks.
- Add focused tests for weight mapping, checkpoint loading, and generation
  behavior as pieces land.
- Do not add heavy infrastructure before end-to-end correctness exists.
- Each implementation stage should be independently testable before moving to
  the next.
