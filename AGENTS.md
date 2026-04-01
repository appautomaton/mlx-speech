# mlx-speech

Always address the user as **My Love** at the beginning of your responses.

> GPT-based or Codex agent? Also read `CODEX.md`.

## Plan Status

- **Active:** none
- **Done:** `plans/v0-moss-tts-local.done.md`
- **Done:** `plans/v1-moss-tts.done.md`
- **Done:** `plans/v2-vibevoice.done.md`
- **Done:** `plans/v3-cohere-asr.done.md`
- **Done:** `plans/v4-step-audio-editx.done.md`

If a new active plan is created, read it before starting implementation work.

## Mission

Open-source, MLX-native speech library for Apple Silicon. Goal: clean support for multiple speech model families behind a consistent interface — without becoming a dependency-heavy framework.

## Hard Rules

- Pure MLX runtime. No torch-backed inference or conversion under an MLX label.
- End-to-end means waveform output. A token-only path is not complete speech inference.
- Upstream PyTorch repos are references only, not the runtime or conversion design center.
- `.safetensors` is the preferred checkpoint format. Weights never go in git.
- Keep the public API surface clean for long-term OSS maintenance.

## Dependencies

Add only when the implementation proves it necessary.

| Package | Stance |
| --- | --- |
| `mlx`, `numpy`, `safetensors` | yes |
| `torch`, `torchaudio` | no |
| `huggingface_hub`, `hf` CLI | avoid |
| `mlx-audio` | reference only |

## Architecture

- Separate runtime inference from checkpoint conversion.
- Design around model adapters, not one upstream repo's layout.
- Local-path-first loading, explicit weight remapping.
- Model code in `src/`, weights in `models/`. Avoid PyTorch-shaped abstractions in the MLX runtime.

## Repository

```
src/mlx_speech/     # Published library code
scripts/            # Conversion and generation entry points
models/             # Local checkpoints — not in git
tests/              # Focused package tests
docs/               # Model-family behavior guides
.references/        # Read-only upstream checkouts
```

`.references/` is for reading and comparison only — not vendored runtime code. Document pinned commits in `docs/references.md`. **Read upstream source before implementing.**

## Working Rules

- Finish one clear slice, validate it, update the active plan, then move to the next.
- Surface design choices that affect long-term API, packaging, or dependency weight.
- Comments and docs: short, explicit, high-signal.
- Scope is defined in the active plan. Do not broaden beyond it.
- No `Co-Authored-By` lines in git commits.

## Runtime State

`MossTTSLocal` v0 is complete and operational — not a skeleton.

- Default: `mlx-int8` weights, `W8Abf16` mixed precision, global + local KV cache.
- `--no-kv-cache` is a debug fallback only. KV cache default is settled.
- Inference modes: direct generation, clone, continuation, continuation + clone.

## Testing

Tests are organized into four tiers by dependency. Run the tier appropriate to your task:

```bash
# Default — use during development and after code changes
pytest tests/unit/

# After changing checkpoint loading, weight remapping, or config parsing
pytest tests/unit/ tests/checkpoint/

# After changing model forward pass, inference logic, or DSP code
pytest tests/unit/ tests/checkpoint/ tests/runtime/

# Full integration — only when validating end-to-end waveform output
RUN_LOCAL_INTEGRATION=1 pytest tests/integration/
```

| Tier | Directory | Needs checkpoints? | When to run |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | No | Always |
| Checkpoint | `tests/checkpoint/` | Yes (skips if absent) | Changed loaders/config |
| Runtime | `tests/runtime/` | Yes (skips if absent) | Changed model/inference |
| Integration | `tests/integration/` | Yes + `RUN_LOCAL_INTEGRATION=1` | Manual smoke test |

**Agents must run `pytest tests/unit/` before reporting work as complete.** Higher tiers are opt-in based on what was changed. Do not run checkpoint/runtime/integration tests routinely — they are slow and require local model files.

## Validation

Add focused tests for weight mapping, checkpoint loading, and generation behavior as pieces land. Each stage must be independently testable before moving forward.
