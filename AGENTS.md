# mlx-speech

Always address the user as **My Love** at the beginning of your responses.

> GPT-based or Codex agent? Also read `CODEX.md`.

## Planning

Do the work in the tree. There is no stage-gated harness.

`.agents/work/` keeps finished change records for bookkeeping. Do not treat
them as an active process, and do not add new ones.

`plans/v0`–`v5*.md` are historical records of shipped work. Do not add to them.

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
| `mlx`, `numpy`, `safetensors`, `soundfile`, `tokenizers` | yes |
| `huggingface_hub` | yes, but lazy — imported only on the weight-download path (`_hub.py`), never for local-path loading |
| `torch`, `torchaudio` | no — conversion and audit scripts only, never the runtime |
| `hf` CLI | avoid |
| `mlx-audio` | reference only |

## Architecture

- Separate runtime inference from checkpoint conversion.
- Design around model adapters, not one upstream repo's layout.
- Local-path-first loading, explicit weight remapping.
- Model code in `src/`, weights in `models/`. Avoid PyTorch-shaped abstractions in the MLX runtime.

## Repository

```
src/mlx_speech/     # Published library code
scripts/            # Conversion, generation, eval, and audit entry points
models/             # Local checkpoints — not in git
tests/              # Focused package tests
docs/               # Model-family behavior guides
.references/        # Read-only upstream checkouts
```

`.references/` is for reading and comparison only — not vendored runtime code. **Read upstream source before implementing.**

## Working Rules

- Finish one clear slice, validate it, then move to the next.
- Surface design choices that affect long-term API, packaging, or dependency weight.
- Comments and docs: short, explicit, high-signal.
- Stay inside the requested change. Do not broaden it.
- No `Co-Authored-By` lines in git commits.

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

Add focused tests for weight mapping, checkpoint loading, and generation behavior as pieces land. Each slice must be testable before moving on.

## Website

The public website is maintained in `sites/mlx-speech/` in
`appautomaton/appautomaton.github.io`. See `WEBSITE.md`; do not recreate a local
website copy or Pages publisher. Model guides remain in `docs/` here.
