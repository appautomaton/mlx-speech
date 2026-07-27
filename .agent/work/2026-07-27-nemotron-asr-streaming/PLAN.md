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
    not block the port itself, only Slice 10.
  - Slice 10 — `human-action`: the Hugging Face publish is outward-facing.
    Confirm before pushing.
- **Execution routes:** all `direct` unless the user asks for subagent
  parallelism. Slices 6 and 7 are the natural candidates if that changes
  (cross-subsystem: conversion + model, and transducer + prompt + tokenizer).

## Weights, worktrees, and skipping gates

This change is being executed in a git worktree. `models/` is gitignored, so a
fresh worktree has **no weights**, and `tests/checkpoint/` + `tests/runtime/`
skip cleanly when checkpoints are absent. Together those two facts let a gate
report success having verified nothing.

Measured in this worktree before the fix: `tests/checkpoint/` gave **38 skipped,
0 passed** with a green exit code, and `tests/unit/` gave 512 passed / 6 skipped
against main's 518 / 0.

Two mitigations, both in place:

1. **Weights are symlinked, not copied.** Each top-level entry under the main
   checkout's `models/` (81 GB) is symlinked into the worktree. `models/*` in
   `.gitignore` has no trailing slash, so symlinks are ignored and `git status`
   stays clean. After linking, the worktree matches main exactly: checkpoint
   32 passed / 6 skipped, unit 518 passed / 0 skipped.
2. **`MLX_SPEECH_REQUIRE_CHECKPOINTS=1` turns any skip into a session failure**
   (`tests/conftest.py`). Every slice gate below must be run with it set. Slice
   evidence must record **pass counts, not skip counts**.

Pre-existing issue found while checking: `models/Qwen3-ASR-1.7B-MLX-BF16` in the
main checkout is a dangling symlink to an empty external directory, so the two
Qwen3-ASR checkpoint gates have been silently skipping. Out of scope here, but it
should be fixed or the tests retired.

## Requirement traceability (SPEC acceptance criteria → slice)

| AC | Criterion | Slice |
| --- | --- | --- |
| AC1 | mel front end matches NeMo featurizer | 2 |
| AC2 | causal subsampling lengths + values match | 3 |
| AC3 | `chunked_limited` mask exact across 5 documented runtime settings | 4 |
| AC4 | converted weights load, every key mapped | 6 |
| AC5 | token-identical transcript, greedy decode | 7 |
| AC6a | streamed == offline encoder, frame-identical (hard gate) | 8 |
| AC6b | streamed == offline **tokens**, ragged chunks + flush (hard gate) | 8 |
| AC13 | live session: arbitrary chunks, persistent state, flush | 8 |
| AC7 | language-specified and `auto` both decode | 7 |
| AC8 | O(n) per-frame work, peak memory, RTFx recorded | 9 |
| AC9 | no torch/NeMo/transformers on inference path | 9 |
| AC10 | `pytest tests/unit/` green | 9 |
| AC11 | int8 build within WER tolerance of bf16 | 10 |
| AC12 | int8 + bf16 published with cards and licenses | 10 |

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

**Status:** complete
**Evidence:** Pinned Hugging Face revision `f3d3333`; upstream listing confirms
both `.nemo` and Transformers formats. Staged the 2.4 GB `.nemo` source at the
planned gitignored path (SHA-256 `210214ed...a74`), extracted 657 fp32 tensors /
638,030,384 parameters, and diffed preprocessor, encoder, decoder, joint, prompt,
and decode limits against DESIGN with no shape/config mismatches. The upstream
repo has no `LICENSE` file, so its model metadata, model card, NVIDIA NGC
governing terms, and the official OpenMDW-1.1 text were checked together; they
identify OpenMDW-1.1 as the governing model license. Official license copy staged
beside the checkpoint (SHA-256 `2ab44b...4df`).
`MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest tests/unit/ -q`:
518 passed, 0 skipped.
**Risks / next:** License decision resolved: converted-weight redistribution is
permitted under OpenMDW-1.1 when the license text and applicable copyright/origin
notices are retained. The `.nemo` config declares four attention contexts while
NVIDIA documents a fifth 160 ms runtime mode; Slice 4 tests all five but does not
mislabel `[56,1]` as config-declared. No human choice remains, so continue to
Slice 2.

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

**Status:** complete
**Evidence:** Added pure-MLX `NemotronFeatureExtractor` with symmetric Hann,
constant-centered STFT, preemphasis, Slaney-area mel filters, `2^-24` log guard,
NeMo valid-frame masking, and explicit `normalize: NA`. Captured a deterministic
NeMo-eval fixture with torch/librosa reference math; runtime source imports
neither. `MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest
tests/unit/test_nemotron_features.py -q`: 9 passed, 0 skipped. Ruff passed.
**Risks / next:** Checkpoint `dither=1e-5` is retained as configuration but not
applied during inference, matching NeMo's `self.training` guard. Slice 3 consumes
the returned valid feature length.

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

**Status:** complete
**Evidence:** Added three-stage causal `dw_striding` subsampling with NeMo list
indices, `(2,1)` asymmetric padding on time and frequency, grouped depthwise +
pointwise stages, vectorized output lengths, valid-frame masking, and `[C,F]`
flatten order. A deterministic torch fixture covers convolution values and weight
layout. `MLX_SPEECH_REQUIRE_CHECKPOINTS=1 .venv/bin/python -m pytest
tests/unit/test_nemotron_subsampling.py -q`: 15 passed, 0 skipped. Ruff passed.
**Risks / next:** none; Slice 4 can proceed independently.

### Slice 4: Relative-position attention + `chunked_limited` mask

**Objective:** Transformer-XL style relative-position MHA with untied per-layer
position biases, plus the cache-aware lookahead mask.
**Acceptance criteria:**
- Mask matches `NeMo/nemo/collections/asr/modules/conformer_encoder.py:856-869`
  exactly, including trunc-division chunk indexing, for all five
  NVIDIA-documented runtime `att_context_size` values. The `.nemo` config
  declares four (`0`, `3`, `6`, `13` right context); NVIDIA's model card also
  documents the 160 ms `1`-frame mode.
- `rel_shift` verified against a hand-computed small case.
- `use_bias=False`; `pos_bias_u` / `pos_bias_v` untied per layer.
- Additive mask (`0` visible, large negative blocked) applied post-scale.
**Verification:** `uv run pytest tests/unit/test_nemotron_attention.py -q`
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/nemotron_asr/attention.py`,
`tests/unit/test_nemotron_attention.py`

**Status:** complete
**Evidence:** Added the exact trunc-division `chunked_limited` additive mask,
Transformer-XL sinusoidal positions, NeMo `rel_shift`, bias-free Q/K/V/out/pos
projections, and untied per-layer `pos_bias_u/v`. Tests cover all five
NVIDIA-documented latency modes, a hand-computed shift, mask visibility, and a
captured torch attention fixture. `MLX_SPEECH_REQUIRE_CHECKPOINTS=1
.venv/bin/python -m pytest tests/unit/test_nemotron_attention.py -q`: 11 passed,
0 skipped. Ruff passed.
**Risks / next:** MLX fused SDPA differs from the unfused torch fixture by at
most `1.28e-4` in the small reference case; Slice 6 activation parity remains
the checkpoint-scale numeric gate.

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

**Objective:** A live streaming session that accepts waveform chunks as they
arrive, holds all cross-chunk state, and produces the same tokens as an offline
decode of the same audio.

**Public session lifecycle** (the API this slice must deliver):

```python
session = model.stream_session(language="en-US", att_context_size=[56, 3])
for pcm in mic:                     # arbitrary sample counts, not hop-aligned
    for token in session.feed(pcm): # may yield nothing for several calls
        ...
tail = session.finalize()           # flush residual samples, mel, and frames
```

**State that must persist across `feed()` calls.** Losing any one of these
silently degrades output while every encoder-level check still passes:

| State | Shape / kind | Consequence if reset per chunk |
| --- | --- | --- |
| Per-layer attention cache | `(1, left_context, 1024)` × 24 | Loses 4.48 s of history |
| Per-layer conv cache | `(1, kernel-1, 1024)` × 24 | Seam artifacts at every boundary |
| Mel cache | bounded frames | Wrong subsampling at boundaries |
| **Residual PCM samples** | `< hop_length` | Dropped/duplicated audio on ragged chunks |
| **RNN-T predictor state** | LSTM `(h, c)` 2×640, `last_token` | Restarts the LM every chunk |

**Acceptance criteria:**
- `stream_session()` accepts chunks of **arbitrary length**, including shorter
  than one mel hop (160 samples) and not aligned to encoder frames. Residual
  samples carry to the next `feed()`.
- Encoder caches preallocated fixed-size and written in place, not
  concatenate-and-slice (see `DESIGN.md`, cache allocation).
- **RNN-T predictor state persists across chunks.** A token emitted in chunk *k*
  conditions decoding in chunk *k+1*. Tested directly, not inferred.
- `finalize()` flushes residual samples, the mel cache, and pending encoder
  frames, emitting any trailing tokens. Without it the tail of the last utterance
  is silently dropped.
- Streaming path uses **no attention mask** — the cached window is the context.
- **Hard gate (AC6a):** streamed encoder output frame-identical to the offline
  `chunked_limited` encoder at native chunk size, within numerical tolerance.
- **Hard gate (AC6b):** cumulative streamed **tokens** equal offline decoded
  tokens for the same waveform fed in ragged chunk sizes (e.g. 1, 137, 4001,
  16000 samples), including the `finalize()` tail. Feeding N chunks equals
  feeding one.
- Both gates must run green, not skip.

AC6a alone is insufficient and was the review's finding: it is satisfiable by an
implementation that preloads the whole recording and chunks it internally, or one
that restarts the decoder each chunk. AC6b plus the ragged-boundary and
predictor-state tests are what make it genuinely live.

**Verification:**
`MLX_SPEECH_REQUIRE_CHECKPOINTS=1 uv run pytest tests/runtime/test_nemotron_streaming.py -q`
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

### Slice 10: Quantize, card, publish

**Objective:** Produce our own MLX int8 and bf16 builds, write model cards, and
publish under `appautomaton`.
**Acceptance criteria:**
- int8 and bf16 builds produced by `scripts/convert/nemotron_asr.py`, following
  the repo's existing quantization path.
- **AC11:** int8 WER on a fixed evaluation set is within an agreed tolerance of
  bf16. Record size reduction and RTFx for both. If int8 costs more than the
  tolerance, it does not ship as the default and the bf16 build leads.
- Streaming hard gate (AC6) re-run against the int8 build. Quantization must not
  break frame identity.
- Model cards in `scripts/hugging_face/model_cards/appautomaton/`, house format,
  carrying the upstream license, NVIDIA attribution, the latency table, and the
  three language tiers stated honestly. A card claiming 40 languages without the
  tier breakdown is misleading.
- **AC12:** `appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx` and
  `-bf16-mlx` live. Naming follows the current convention (`-int8-mlx`,
  `-bf16-mlx`), not the older `-8bit-mlx` form.
- `_hub` resolver and `mlx_speech.asr.load(...)` aliases default to the published
  int8 repo, mirroring `qwen3-asr`.
- README model table updated.
**Verification:** `hf auth whoami`; repo listing; `MLX_SPEECH_REQUIRE_CHECKPOINTS=1
uv run pytest tests/unit/ tests/runtime/test_nemotron_streaming.py -q`
**Execution:** direct
**Depends on:** Slice 9
**Checkpoint after:** human-action (the publish is outward-facing)
**Touches:** `scripts/convert/nemotron_asr.py`, `scripts/hugging_face/upload.py`,
`scripts/hugging_face/model_cards/appautomaton/`, `src/mlx_speech/_hub.py`,
`README.md`, `docs/nemotron-asr.md`

**Status:** pending

## Aggregate verification

Every command below assumes `MLX_SPEECH_REQUIRE_CHECKPOINTS=1` is set, so a skip
fails the run. Evidence records pass counts, never skip counts.

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
| 10 | `hf auth whoami`; repo listing; unit + streaming gate against the int8 build |

## Riskiest slice

Slice 8. Cache-aware state is the one subsystem with no precedent in this repo,
and it has two independent failure modes that present differently. A correctness
bug shows up in the frame-identity gate. A performance bug — reverting to
buffered recomputation — does not, and passes every correctness test while
throwing away the entire reason for choosing this model.

Slice 7 is the runner-up: RNN-T is a new decoder family here, and a subtly wrong
greedy loop produces fluent, plausible, incorrect transcripts.

## Review: Engineering

- Verdict: needs_correction
- Strength: The plan decomposes the pure-MLX port into reference-anchored, independently testable slices and makes frame identity, runtime purity, and no-skip gates explicit.
- Concern: Slice 8 specifies encoder caches but not a public waveform-chunk session, final-tail flush, persistent RNN-T decoder state, or end-to-end streamed-versus-offline token parity, so an implementation that preloads the recording or restarts decoding per chunk could satisfy the written gates without providing true live streaming.
- Action: Define the public streaming session lifecycle and state in Slice 8, then add arbitrary waveform-boundary and final-flush tests that require cumulative streamed tokens to equal offline decoding while preserving predictor state across chunks.
- Verified: PLAN, DESIGN, SPEC, current ASR protocol and registry, mlx-audio streaming/Conformer/RNN-T reference, NeMo mask and featurizer source, checkpoint skip enforcement, and 518 passing unit tests were checked.

### Response (auto-plan, 2026-07-27)

Concern upheld in full. AC6 as written was satisfiable by an implementation that
preloads the recording and chunks internally, or restarts the decoder each chunk.
Encoder frame identity does not imply a live session.

Slice 8 rewritten with:
- a concrete public `stream_session()` / `feed()` / `finalize()` lifecycle;
- an explicit table of the five pieces of cross-chunk state, including the two
  the plan previously omitted — residual sub-hop PCM samples and RNN-T predictor
  state (LSTM `h`/`c` plus `last_token`);
- `finalize()` tail flush as an acceptance criterion, not an implementation
  detail;
- AC6 split into AC6a (encoder frame identity) and AC6b (cumulative streamed
  tokens equal offline tokens at ragged, non-hop-aligned chunk boundaries).

SPEC AC6 updated to match; new AC13 covers the session lifecycle.

Also recorded from slice-1 investigation, which contradicts the current plan text
and must be reconciled during execution: the published `config.json` reports
`sliding_window: 57` and `supported_num_lookahead_tokens: [3, 0, 6, 13]` with
`default_num_lookahead_tokens: 3` — four trained lookahead values, not the five
this plan and DESIGN.md currently document, and a left context of 57 rather
than 56.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Slice 8 now defines a true live-audio session, enumerates every cross-chunk state component, and hard-gates both encoder-frame identity and ragged-chunk token parity through finalization.
- Concern: The remaining integration risk is that Slice 8's write set names only model files even though users obtain Nemotron through `mlx_speech.asr.load()`, whose current `ASRModel` protocol and adapters expose only `generate()`.
- Action: Expose and test `stream_session()` through the Nemotron ASR adapter during Slice 8 or Slice 9 before marking AC13 complete.
- Verified: Revised PLAN and SPEC, DESIGN, current ASR protocol and adapters, pinned mlx-audio streaming implementation, NeMo references, NVIDIA's documented five latency modes, and 518 passing unit tests were checked.
