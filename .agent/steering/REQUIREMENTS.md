# Requirements

Accepted product and technical constraints. These are hard rules, not
preferences. A change that violates one is wrong even if it works.

## Runtime purity

- **Pure MLX.** No torch-backed inference or conversion under an MLX label. If a
  path routes through PyTorch, Transformers, vLLM, ONNX, or an upstream runtime,
  it is not shippable here.
- **End-to-end means waveform.** A token-only path is not complete speech
  inference. TTS produces audio. ASR produces text from audio.
- **Upstream repos are references, not the design center.** Read them as a
  specification. Do not let one upstream repo's layout dictate the architecture.

## Checkpoints and weights

- `.safetensors` is the preferred format.
- **Weights never go in git.** They live under `models/`, which is gitignored.
- Local-path-first loading with explicit weight remapping. No silent key
  fallbacks, no quietly dropped tensors.
- Published weights go to `appautomaton` on Hugging Face with a license file and
  upstream attribution. Publishing is outward-facing and requires explicit
  confirmation.

## Dependencies

| Package | Stance |
| --- | --- |
| `mlx`, `numpy`, `safetensors` | yes |
| `torch`, `torchaudio` | no |
| `huggingface_hub`, `hf` CLI | avoid |
| `mlx-audio` | reference only |

Add anything else only when the implementation proves it necessary.

## Architecture

- Separate runtime inference from checkpoint conversion.
- Design around model adapters, not one upstream repo's layout.
- Model code in `src/mlx_speech/models/`, entry points in `scripts/`, focused
  tests in `tests/`, behavior guides in `docs/`.
- `.references/` holds read-only upstream checkouts. Never vendored, never
  imported by the runtime, pinned by commit in `docs/references.md`.
- Avoid PyTorch-shaped abstractions in the MLX runtime.

## Testing

Four tiers by dependency:

| Tier | Directory | Needs checkpoints? | When |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | No | Always |
| Checkpoint | `tests/checkpoint/` | Yes, skips if absent | Changed loaders or config |
| Runtime | `tests/runtime/` | Yes, skips if absent | Changed model or inference |
| Integration | `tests/integration/` | Yes + `RUN_LOCAL_INTEGRATION=1` | Manual smoke test |

`pytest tests/unit/` must pass before any work is reported complete. Higher tiers
are opt-in based on what changed. CI runs unit and checkpoint on `macos-latest`,
since MLX has no x86 CPU fallback.

A parity gate that can skip when fixtures are absent is not a gate. Make it hard
or do not claim it.

**Running gates:** set `MLX_SPEECH_REQUIRE_CHECKPOINTS=1`. Any skip then fails the
session and names what was skipped (`tests/conftest.py`). Slice evidence records
pass counts, never skip counts. Without it, a checkpoint or runtime tier with no
weights present exits green having verified nothing.

## Worktrees and weights

`models/` is gitignored and holds tens of GB. A fresh git worktree therefore has
**no weights**, which silently converts every checkpoint and runtime gate into a
no-op.

Symlink, never copy. For each top-level entry in the main checkout's `models/`:

```bash
for d in /path/to/main/models/*/; do ln -sfn "$d" "models/$(basename "$d")"; done
```

`models/*` in `.gitignore` has no trailing slash, so it ignores symlinks as well
as directories and `git status` stays clean. Verify with `git check-ignore -v`
and by confirming the worktree's test counts match the main checkout's.

The same applies to `.references/`. Its ignore pattern was corrected to
`.references` (no trailing slash) for exactly this reason: a trailing slash
matches directories only, so a symlinked reference checkout would have been
committed.

## Process

- Finish one clear slice, validate it, update the plan, then move to the next.
- Surface design choices that affect long-term API, packaging, or dependency
  weight.
- Comments and docs stay short, explicit, and high-signal.
- Scope is defined by the active change. Do not broaden beyond it.
- No `Co-Authored-By` lines in commits.
