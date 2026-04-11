# Clone Eval Set

This directory defines the fixed v1 voice-clone evaluation surface.

Current source:

- `macos_builtin_en.json` — deterministic English eval set generated from
  macOS built-in voices via `say`

The committed files here are metadata only. Reference audio is generated
locally, not committed.

Materialize the reference audio:

```bash
uv run python scripts/eval/materialize_clone_eval_macos.py \
  --manifest examples/clone_eval/macos_builtin_en.json \
  --output-dir outputs/clone_eval/macos_builtin_en
```

Run a clone preset sweep:

```bash
uv run python scripts/eval/sweep_clone_presets.py \
  --manifest examples/clone_eval/macos_builtin_en.json \
  --reference-dir outputs/clone_eval/macos_builtin_en/references \
  --output-dir outputs/clone_eval/macos_builtin_en/runs
```

The v1 quality scope for this set is:

- single clean reference
- same-language clone
- English only
