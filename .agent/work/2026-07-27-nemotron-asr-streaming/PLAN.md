# PLAN: Nemotron 3.5 ASR streaming (pure-MLX)

**Goal:** Port NVIDIA Nemotron 3.5 ASR (cache-aware streaming FastConformer-RNNT,
40 language-locales) to a pure-MLX runtime with both offline and true streaming
paths. Full contract: `SPEC.md`. Architecture: `DESIGN.md`.

## Execution routing and topology

- **Default:** continuation. After a slice verifies, the next begins. Execution
  windows are context batches, not stopping points.
- **Parallel-safe groups:** {Slice 2, Slice 3, Slice 4} — disjoint write sets
  (`feature_extraction.py` vs `subsampling.py` vs `attention.py`), all depend
  only on Slice 1. All three are structural and testable without weights.
- **Hard gate:** Slice 8. Streamed encoder output must be frame-identical to the
  offline encoder at native chunk size. This is an architectural invariant, not
  an approximation, and it needs no external reference. It must run green, not
  skip. Slice 9 depends on Slice 8, so an implementation that transcribes
  plausibly but has silently reverted to buffered streaming cannot ship.
- **Serial spine:** 1 → (2,3,4) → 5 → 6 → 7 → 8 → 9.
- **Checkpoints:**
  - Slice 1 — `decision`: the OpenMDW-1.1 vs NVIDIA Open Model License question
    must be resolved from the LICENSE file before any redistribution work. Does
    not block the port itself.
- **Execution routes:** all `direct` unless the user asks for subagent
  parallelism. Slices 6 and 7 are the natural candidates if that changes
  (cross-subsystem: conversion + model, and transducer + prompt + tokenizer).

## Requirement traceability (SPEC acceptance criteria → slice)

| AC | Criterion | Slice |
| --- | --- | --- |
| AC1 | mel front end matches NeMo featurizer | 2 |
| AC2 | causal subsampling lengths + values match | 3 |
| AC3 | `chunked_limited` mask exact across 5 settings | 4 |
| AC4 | converted weights load, every key mapped | 6 |
| AC5 | token-identical transcript, greedy decode | 7 |
| AC6 | streamed == offline, frame-identical (hard gate) | 8 |
| AC7 | language-specified and `auto` both decode | 7 |
| AC8 | O(n) per-frame work, peak memory, RTFx recorded | 9 |
| AC9 | no torch/NeMo/transformers on inference path | 9 |
| AC10 | `pytest tests/unit/` green | 9 |

## Slices

### Slice 1: Confirm checkpoint format, license, and reference parity

**Objective:** Resolve the two open questions in `DESIGN.md` before any code:
what the HF repo actually ships, and which license governs it. Stage weights
locally and record the reference commit set.
**Acceptance criteria:**
- Checkpoint format confirmed (`.nemo` tarball vs transformers layout vs both),
  with the actual file list recorded.
- LICENSE file read and the OpenMDW-1.1 / NVIDIA Open Model License question
  settled in writing.
- Weights staged at `models/nvidia/nemotron_3_5_asr_streaming_0_6b/original/`
  (gitignored).
- `config.json` / `model_config.yaml` values cross-checked against the
  `ConformerArgs` / `PredictArgs` / `JointArgs` in `DESIGN.md`.
**Verification:** file listing printed; key and parameter counts printed; config
values diffed against DESIGN tables.
**Execution:** direct
**Depends on:** none
**Checkpoint after:** decision (license)
**Touches:** `docs/references.md`, `models/` (gitignored)

**Status:** pending

### Slice 2: Mel front end (NeMo parity)

**Objective:** 128-mel featurizer matching NeMo `AudioToMelSpectrogramPreprocessor`.
**Acceptance criteria:**
- `n_fft` 512, win 400, hop 160, `preemph` 0.97, `dither` 1e-5,
  `log_zero_guard_value` 2^-24, Hann window.
- **`normalize: NA`** — no per-feature normalization. Explicitly tested, since
  this is the silent-failure mode.
- Output matches a NeMo reference mel on a fixed waveform within tolerance.
**Verification:** `uv run pytest tests/unit/test_nemotron_features.py -q`
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/nemotron_asr/feature_extraction.py`,
`tests/unit/test_nemotron_features.py`

**Status:** pending

### Slice 3: Causal depthwise-striding subsampling (8x)

**Objective:** Three stride-2 stages reducing 100 Hz to 12.5 Hz, with causal
asymmetric padding.
**Acceptance criteria:**
- Padding is `left = kernel - 1 = 2`, `right = stride - 1 = 1`, applied on **both**
  time and frequency axes.
- Output length matches NeMo's `_calc_length` recurrence for a range of inputs.
- Channel-major flatten order `(B, T', F', C) → (B, T', C·F')` matches NeMo.
- First stage is a full conv; stages 2 and 3 are depthwise + pointwise.
**Verification:** `uv run pytest tests/unit/test_nemotron_subsampling.py -q`
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/nemotron_asr/subsampling.py`,
`tests/unit/test_nemotron_subsampling.py`

**Status:** pending

### Slice 4: Relative-position attention + `chunked_limited` mask

**Objective:** Transformer-XL style relative-position MHA with untied per-layer
position biases, plus the cache-aware lookahead mask.
**Acceptance criteria:**
- Mask matches `NeMo/nemo/collections/asr/modules/conformer_encoder.py:856-869`
  exactly, including trunc-division chunk indexing, for all five trained
  `att_context_size` values.
- `rel_shift` verified against a hand-computed small case.
- `use_bias=False`; `pos_bias_u` / `pos_bias_v` untied per layer.
- Additive mask (`0` visible, large negative blocked) applied post-scale.
**Verification:** `uv run pytest tests/unit/test_nemotron_attention.py -q`
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/nemotron_asr/attention.py`,
`tests/unit/test_nemotron_attention.py`

**Status:** pending

### Slice 5: FastConformer encoder assembly

**Objective:** 24-layer encoder composing Slices 2–4.
**Acceptance criteria:**
- Macaron block order: half-FFN, attention, causal conv, half-FFN, final norm.
- Conv module uses LayerNorm under the attribute name `batch_norm`.
- Causal depthwise conv, kernel 9, left-pad 8 / right-pad 0.
- Forward runs on a fixed input and produces the expected shape and finite values.
**Verification:** `uv run pytest tests/unit/test_nemotron_encoder.py -q`
**Execution:** direct
**Depends on:** Slices 2, 3, 4
**Touches:** `src/mlx_speech/models/nemotron_asr/encoder.py`,
`src/mlx_speech/models/nemotron_asr/config.py`,
`tests/unit/test_nemotron_encoder.py`

**Status:** pending

### Slice 6: Checkpoint conversion and loading

**Objective:** Explicit remap from NeMo keys to the MLX layout, no silent
fallbacks.
**Acceptance criteria:**
- Every source key maps to exactly one destination. None missing, none extra,
  none silently dropped. Unmapped keys raise.
- Conv weight layout transposed correctly for MLX channels-last.
- `vocabulary` and `prompt_dictionary` extracted.
- Encoder activations match the mlx-audio reference on a fixed input.
**Verification:** `uv run pytest tests/checkpoint/test_nemotron_load.py -q`
**Execution:** direct
**Depends on:** Slice 5
**Touches:** `src/mlx_speech/models/nemotron_asr/checkpoint.py`,
`scripts/convert/nemotron_asr.py`, `tests/checkpoint/test_nemotron_load.py`

**Status:** pending

### Slice 7: RNN-T decode and language prompt — first transcript

**Objective:** Prediction network, joint network, greedy transducer decode, and
language-ID prompt fusion. First end-to-end transcript.
**Acceptance criteria:**
- Prediction net: embedding over `vocab + 1` (blank as pad) into a 2-layer LSTM
  at 640 hidden.
- Joint: `enc 1024→640`, `pred 640→640`, ReLU, `640→13088`. One lattice cell at a
  time; the T×U lattice is never materialized.
- Greedy loop: blank advances time, non-blank advances text, `max_symbols = 10`
  guard.
- Prompt: one-hot 128 concatenated on the feature axis, `1152 → 2048 → ReLU →
  1024`.
- Transcript is token-identical to the reference on a fixed clip (AC5).
- Language-specified and `auto` both decode correctly (AC7).
**Verification:** `uv run pytest tests/runtime/test_nemotron_decode.py -q`
**Execution:** direct
**Depends on:** Slice 6
**Touches:** `src/mlx_speech/models/nemotron_asr/transducer.py`,
`src/mlx_speech/models/nemotron_asr/prompt.py`,
`src/mlx_speech/models/nemotron_asr/tokenizer.py`,
`src/mlx_speech/models/nemotron_asr/model.py`,
`tests/runtime/test_nemotron_decode.py`

**Status:** pending

### Slice 8: Cache-aware streaming — HARD GATE

**Objective:** Per-layer attention and conv caches with incremental subsampling,
giving O(n) streaming with latency independent of utterance length.
**Acceptance criteria:**
- Attention cache holds the last `left_context` attention inputs; conv cache
  holds the last `kernel - 1` post-GLU frames. Both **preallocated fixed-size**
  and written in place, not concatenate-and-slice (see `DESIGN.md`, cache
  allocation).
- Incremental subsampling with a bounded mel cache.
- Streaming path uses **no attention mask** — the cached window is the context.
- **Hard gate (AC6):** streamed encoder output frame-identical to the offline
  `chunked_limited` encoder at native chunk size (`right_context + 1`), within
  numerical tolerance. Must run green, not skip.
- Streaming entry point yields incremental transcripts.
**Verification:** `uv run pytest tests/runtime/test_nemotron_streaming.py -q`
(hard gate: must run green, not skip)
**Execution:** direct
**Depends on:** Slice 7
**Touches:** `src/mlx_speech/models/nemotron_asr/streaming.py`,
`src/mlx_speech/models/nemotron_asr/model.py`,
`tests/runtime/test_nemotron_streaming.py`

**Status:** pending

### Slice 9: Performance validation, purity, and docs

**Objective:** Prove the streaming path is actually O(n) and competitive, confirm
runtime purity, and document.
**Acceptance criteria:**
- **O(n) check (AC8):** per-frame work is constant as audio length grows.
  Correctness tests do not catch a regression to buffered streaming; this does.
- Peak memory (`mx.get_peak_memory()`) and RTFx recorded against the mlx-audio
  reference. Slower than the reference is a defect, not a tradeoff.
- **Purity (AC9):** no torch, NeMo, or transformers import on the inference path.
- `mlx_speech.asr.load(...)` alias registered, local-path-first.
- `docs/nemotron-asr.md` written: the two independent knobs, the three language
  tiers stated honestly, latency table, streaming usage.
- **AC10:** `uv run pytest tests/unit/` green.
**Verification:** `uv run pytest tests/unit/ tests/runtime/test_nemotron_purity.py -q`;
benchmark output recorded in the slice evidence.
**Execution:** direct
**Depends on:** Slice 8
**Touches:** `src/mlx_speech/asr/`, `docs/nemotron-asr.md`, `README.md`,
`tests/runtime/test_nemotron_purity.py`, `tests/unit/`

**Status:** pending

## Aggregate verification

| Slice | Command |
| --- | --- |
| 1 | file listing + key/param counts printed; LICENSE read |
| 2 | `uv run pytest tests/unit/test_nemotron_features.py -q` |
| 3 | `uv run pytest tests/unit/test_nemotron_subsampling.py -q` |
| 4 | `uv run pytest tests/unit/test_nemotron_attention.py -q` |
| 5 | `uv run pytest tests/unit/test_nemotron_encoder.py -q` |
| 6 | `uv run pytest tests/checkpoint/test_nemotron_load.py -q` |
| 7 | `uv run pytest tests/runtime/test_nemotron_decode.py -q` |
| 8 | `uv run pytest tests/runtime/test_nemotron_streaming.py -q` (hard gate) |
| 9 | `uv run pytest tests/unit/ tests/runtime/test_nemotron_purity.py -q` + benchmark |

## Riskiest slice

Slice 8. Cache-aware state is the one subsystem with no precedent in this repo,
and it has two independent failure modes that present differently. A correctness
bug shows up in the frame-identity gate. A performance bug — reverting to
buffered recomputation — does not, and passes every correctness test while
throwing away the entire reason for choosing this model.

Slice 7 is the runner-up: RNN-T is a new decoder family here, and a subtly wrong
greedy loop produces fluent, plausible, incorrect transcripts.
