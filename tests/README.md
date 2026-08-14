# Testing

Tests protect existing runtime behavior. Reorganizing tests, fixtures, CI, or
coverage does not authorize changes under `src/`. If test cleanup exposes a
production defect, report it and fix it in a separate behavior change.

## Tiers

| Tier | Purpose | Local artifacts | Default CI |
| --- | --- | --- | --- |
| `unit/` | Pure logic, tiny MLX models, synthetic checkpoints, and bounded oracle fixtures | No | Yes |
| `checkpoint/` | Loading and alignment against real local checkpoint assets | Yes | Collection only |
| `runtime/` | Real-weight forward, inference, streaming, and model-level parity | Yes | Collection only |
| `integration/` | Public API through waveform or transcript output | Yes, plus evaluation inputs | Collection only |

The default command runs only the fast tier:

```bash
pytest
```

Run an opt-in tier by naming its directory:

```bash
pytest tests/checkpoint/
pytest tests/runtime/
RUN_LOCAL_INTEGRATION=1 pytest tests/integration/
```

When an opt-in run is acting as a required gate, make skips fail the session:

```bash
MLX_SPEECH_REQUIRE_CHECKPOINTS=1 pytest tests/runtime/
```

Use this inexpensive command to validate imports and collection across every
tier without running model inference:

```bash
pytest --collect-only -q tests/unit/ tests/checkpoint/ tests/runtime/ tests/integration/
```

Do not put `test_*.py` files directly under `tests/`. Tier directories are
Python packages so identically named tests in different tiers cannot collide
during combined collection.

## Placement rules

A test belongs in `unit/` only when it is deterministic and needs no network,
local model directory, upstream checkout, or optional Torch environment. Prefer
real tiny MLX modules and tiny safetensors over mocks.

Checkpoint tests validate real artifact layout, keys, shapes, storage/runtime
dtypes, quantization metadata, and strict alignment. Pure remapping and loader
branches still belong in `unit/` and should use synthetic checkpoint files.

Runtime tests load real weights and exercise component or inference behavior.
Integration tests cross the public API boundary and must reach waveform or
transcript output. A finite, non-empty waveform assertion is a smoke test, not
a quality gate; WER, CER, speaker similarity, numeric parity, memory, and timing
regressions need their own explicit metrics.

Every bug fix adds the smallest test that fails before the fix. Put that test in
the lowest tier capable of reproducing the bug, then add a higher-tier contract
test only when the failure can cross a component boundary.

## Test doubles

Use stubs or fakes at expensive and nondeterministic boundaries: Hub access,
network calls, subprocesses, clocks, file writers, tokenizers, and heavyweight
model adapters. A test double should reject unexpected calls and preserve the
boundary's input/output contract.

Do not replace the behavior under test. Loader tests use real tiny files;
generation-state tests use a tiny model or a narrow adapter fake; numeric model
tests run the real layer implementation.

## Golden fixtures

Golden fixtures isolate the MLX test from the upstream reference environment.
They avoid a full checkpoint only when the fixture contains the required small
weights or the tested component is weight-free.

Committed oracle fixtures must record:

- deterministic input construction and seed;
- reference repository revision and dependency versions;
- array shapes, dtypes, hashes, and bounded file sizes;
- a tolerance chosen for the numeric quantity being compared;
- capture and regeneration commands.

Use exact equality for discrete schedules and tokens, `allclose` for stable
tensor math, and scale-aware metrics such as correlation or relative RMSE for
waveforms and complex spectra. Do not regenerate a fixture merely because a
test failed. Review the behavioral difference and the pinned reference first.

## Coverage

CI measures line and branch coverage for `mlx_speech`. The floor in
`pyproject.toml` is the measured fast-suite baseline, not a claim that every
model family has sufficient behavioral coverage. It may move upward with new
tests and must not be lowered to make CI pass.

New or changed production behavior needs direct coverage for success, boundary,
and failure paths. Model-family reviews should trace config parsing, checkpoint
mapping, component parity, generation state, real runtime inference, public API
output, and quality/performance gates instead of relying on a single aggregate
percentage.
