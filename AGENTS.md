# mlx-speech

MLX-native speech library for Apple Silicon: TTS, ASR, and alignment model
families behind one small, consistent API, without a heavy dependency footprint.

> GPT-based or Codex agent? Also read `CODEX.md`.

## Hard rules

- The runtime is pure MLX. `torch` may appear only in conversion, audit, and
  `dev/parity/` scripts, never in anything `src/mlx_speech/` imports.
- End-to-end means a waveform for TTS and a transcript for ASR. A token-only
  path is not finished inference.
- Upstream repos are references, not the design center. Read the upstream
  source in `.references/` before implementing; never import or vendor it.
- Checkpoints are `.safetensors` under `models/`. Weights never go in git.
- The public API is long-lived OSS surface. Raise any change to it, to
  packaging, or to dependencies before making it.

## Design

- Design around model adapters, not one upstream repo's layout. No
  PyTorch-shaped abstractions in the runtime.
- Keep conversion separate from runtime inference. Remap weights explicitly.
- Loading is local-path first. A local path must never import or reach the Hub.
- Gate every `mx.fast.metal_kernel` path on `MLX_SPEECH_DISABLE_CUSTOM_METAL=1`
  with a pure-MLX fallback. CI sets it because GitHub's macOS runner reports
  Metal but aborts custom kernels. Pattern: `models/dots_tts/vocoder.py`.

## Dependencies

Add one only when the implementation proves it necessary.

| Package | Stance |
| --- | --- |
| `mlx`, `numpy`, `safetensors`, `soundfile`, `tokenizers` | Runtime |
| `huggingface_hub` | Runtime, imported only inside `_hub.py` download functions |
| `torch`, `torchaudio` | Conversion, audit, and parity scripts only |
| `hf` CLI | Release workflow only; library code never shells out to it |
| `mlx-audio` | Reference only |

## Layout

```
src/mlx_speech/
  tts/ asr/ align/           # Public API: one adapter and registry entry per family
  generation/<family>.py     # Inference loops
  models/<family>/           # Architecture, config, checkpoint loading and remapping
  _hub.py                    # Model catalog and lazy download
scripts/convert/<family>.py  # Upstream checkpoint -> MLX safetensors
scripts/                     # Also generate, eval, audit, release, hugging_face
dev/parity/                  # Golden-tensor capture from upstream (dev only)
tests/                       # Tiered; rules in tests/README.md
docs/<family>.md             # Model guides, indexed in docs/README.md
models/  .references/        # Local only, gitignored
```

A new family usually touches every layer above, plus `README.md`. Commit
`210e418` (Qwen3-ForcedAligner) is a complete example.

## Testing

Run tools through `uv`. `uv run pytest` runs only `tests/unit/`; the other tiers
run only when named. Unit tests must pass before any work is reported complete.

| Changed | Also run |
| --- | --- |
| Loading, weight remapping, config | `tests/checkpoint/` |
| Forward pass, inference, DSP, kernels | `tests/checkpoint/ tests/runtime/` |
| End-to-end output (manual smoke) | `RUN_LOCAL_INTEGRATION=1 uv run pytest tests/integration/` |

Opt-in tiers skip when local checkpoints are missing, and a skipped run is not a
pass. When a tier is your gate, set `MLX_SPEECH_REQUIRE_CHECKPOINTS=1` so skips
fail, and report what did not run.

Every bug fix adds the smallest test that fails before the fix. Never lower the
coverage floor in `pyproject.toml`. Keep `uv run ruff check` clean. Test
placement, doubles, fixtures, and tolerances: `tests/README.md`.

## Working rules

- Take scope from the user. Finish one slice, validate it, then move on.
- Stay inside the requested change.
- Comments and docs: short, explicit, high-signal.
- No `Co-Authored-By` lines in commits.
- Parallel agent sessions use git worktrees under `.worktrees/` (gitignored).

## Records

`.agents/` and `plans/` hold finished history. Nothing there is active. Do not
execute, extend, or add to them.

## Website

The public site lives in `sites/mlx-speech/` of
`appautomaton/appautomaton.github.io`; see `WEBSITE.md`. Do not recreate it
here. Model guides stay in `docs/`.
