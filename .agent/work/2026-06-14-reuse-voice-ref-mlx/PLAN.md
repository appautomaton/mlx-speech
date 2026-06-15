# PLAN: RE-USE voice-reference denoising (pure-MLX)

**Goal:** Port NVIDIA RE-USE (SEMamba) to pure MLX and wire it into DramaBox
`denoise_ref=True` (opt-in). Full contract: `SPEC.md`. Design: `DESIGN.md`.

## Execution routing and topology

- **Default:** continuation. After a slice verifies, the next begins. Execution
  windows are context batches, not stopping points.
- **Parallel-safe groups:** {Slice 2, Slice 3} — disjoint write sets
  (`mamba/` vs `stft.py`), both depend only on Slice 1.
- **Hard gate:** Slice 6 (torch-fixture parity, AC3) is the only end-to-end
  numeric proof. Integration (Slice 7) depends on Slice 6, not Slice 5, so a
  self-consistent but numerically wrong enhancer cannot reach DramaBox. Serial
  spine: 5 -> 6 -> 7.
- **Checkpoints:**
  - Slice 6 — `human-action`: parity fixtures may need a host with
    `torch` + `mamba_ssm` (no macOS wheels). Pause only if they cannot be
    generated locally.
  - Slice 8 — `human-action`: the Hugging Face publish is outward-facing;
    confirm before pushing.
- **Subagent routes:** Slice 4 and Slice 7 are `subagent recommended`
  (cross-subsystem: model+conversion, and generate+hub+adapter).

## Requirement traceability (SPEC acceptance criteria → slice)

| AC | Criterion | Slice |
| --- | --- | --- |
| AC1 | selective-scan matches reference | 2 |
| AC2 | converted weights load, all keys mapped | 4 |
| AC3 | parity thresholds vs torch reference | 6 |
| AC4 | `denoise_ref=True` finite stereo, differs, cached | 7 |
| AC5 | `denoise_ref=False` unchanged (regression) | 7 |
| AC6 | no torch on the runtime path | 5, 7 |
| AC7 | `pytest tests/unit/` green | 8 |
| AC8 | appautomaton repo carries NSCLv1 + attribution | 8 |

## Slices

### Slice 1: Acquire and document the RE-USE reference

**Objective:** Vendor `nvidia/RE-USE` code + config into `.references/` at a
pinned commit, download SEMamba weights locally, vendor the `mamba_ssm`
`selective_scan_ref` source (the exact reference math Slice 2 must mirror, since
the package has no macOS wheels), and write an architecture note (module
inventory, STFT config, bidirectional combine, weight-key list).
**Acceptance criteria:**
- `.references/RE-USE/` present; pinned commit recorded in `docs/references.md`.
- The `selective_scan_ref` reference source is vendored under `.references/` and cited in the notes file (Slice 2 mirrors it, not an independent re-derivation).
- Weights downloaded to `models/reuse/original/` (gitignored).
- A notes file enumerates SEMamba submodules, the config (n_fft/hop/win, compress_factor, block count, bidirectional combine rule, dt_bias/A/B/C/D layout), and the checkpoint key list with shapes.
**Verification:** `uv run python scripts/convert/reuse_inspect.py --checkpoint models/reuse/original` prints key count + total params (≈9.6M); `test -d .references/RE-USE` and the vendored `selective_scan_ref` source exists.
**Touches:** `.references/RE-USE/`, `models/reuse/original/`, `scripts/convert/reuse_inspect.py`, `docs/references.md`, `.agent/work/2026-06-14-reuse-voice-ref-mlx/slices/slice-001-notes.md`
**Produces:** vendored reference + `selective_scan_ref` source + key/shape inventory used by Slices 2-4.

**Status:** complete
**Evidence:** vendored `.references/RE-USE/` (SEMamba code, config, recipe yaml, chunked inference) and `.references/mamba_ssm/` (`selective_scan_ref` + `mamba_simple`, `v2.2.2`/`8ffd905`); weights at `models/reuse/original/model.safetensors` (gitignored, 37 MB); `scripts/convert/reuse_inspect.py` prints `1416 keys / 9.61M params`; architecture inventory at `slices/slice-001-notes.md` (bidirectional combine rule + per-direction Mamba layout + dt/A/B/C/D); pins in `docs/references.md`. RE-USE pinned at `7619050`. Verification green.
**Risks / next:** none. Slice 2 mirrors `.references/mamba_ssm/selective_scan_interface.py:selective_scan_ref`; Slice 3 mirrors `.references/RE-USE/models/stfts.py`.

### Slice 2: Selective-scan primitive in MLX

**Objective:** Implement the SSM selective scan (forward + bidirectional) in MLX,
validated against a numpy port of `mamba_ssm`'s `selective_scan_ref` (vendored in
Slice 1), not an independently invented recurrence.
**Acceptance criteria:**
- MLX scan matches the `selective_scan_ref`-derived numpy reference within tol on random inputs.
- Cases explicitly cover: `dt` softplus + `dt_bias`, the A exponent sign, input-dependent (variable) B/C layout, the D skip connection, and the reverse-branch scan plus the bidirectional combine rule from Slice 1's notes.
- Output shapes match `(B, T, ...)` expectations; runs with no torch.
**Verification:** `uv run pytest tests/unit/test_reuse_scan.py -q`
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/reuse/mamba/scan.py`, `tests/unit/test_reuse_scan.py`

**Status:** complete
**Evidence:** `scan.py` (`selective_scan` + `selective_scan_reverse`) mirrors `selective_scan_ref` (variable B/C, D skip, silu(z) gate, softplus + delta_bias); 9 tests vs a numpy float64 port of the reference, max abs diff < 1e-4; reuse unit tests 22 passed; no torch in source. Spec + quality review APPROVED (line-by-line parity confirmed against the vendored reference, with anti-no-op controls).
**Risks / next:** MLX 0.31.1 mishandles elementwise ops on reversed-stride views and lacks `mx.flip` — neutralized here by an `mx.contiguous` guard + negative-step slicing. Carry this caveat into Slice 4 if it flips around elementwise ops.

### Slice 3: STFT front end (mag/phase + iSTFT + sweep filter)

**Objective:** Port `mag_phase_stft` / `mag_phase_istft` with `compress_factor`,
the sweep-artifact filter, and chunked Hann overlap-add to MLX.
**Acceptance criteria:**
- `iSTFT(STFT(x)) ≈ x` within tol (round-trip).
- n_fft/hop/win scale from the config training rate as in the reference.
- Sweep-artifact filter reproduces reference behavior on a crafted input.
**Verification:** `uv run pytest tests/unit/test_reuse_stft.py -q`
**Depends on:** Slice 1
**Touches:** `src/mlx_speech/models/reuse/stft.py`, `tests/unit/test_reuse_stft.py`

**Status:** complete
**Evidence:** `stft.py` (`mag_phase_stft`/`mag_phase_istft` on `mx.fft`, `relu_log1p` compress/expand, `sweep_artifact_filter`, `chunked_hann_ola`, `stft_params_for`) mirrors `.references/RE-USE/models/stfts.py` + `super_resolution.py:224-265`; 13 tests, round-trip interior ~5e-6 at op_sr 8k/16k. Spec + quality review APPROVED.
**Risks / next:** non-blocking quality follow-ups recorded — dedupe the window-pad block into a `_padded_hann` helper, and add a compress-param-overload docstring note. Apply during a later cleanup pass.

### Slice 4: SEMamba assembly + weight conversion

**Objective:** Build the MLX SEMamba module (Mamba blocks, dense connections,
heads) and `scripts/convert/reuse.py` mapping `nvidia/RE-USE` weights to MLX
`.safetensors`.
**Acceptance criteria:**
- Converted weights load into the MLX module with every key mapped (no missing/extra).
- Forward on a dummy STFT input returns correct-shaped `(amp_g, pha_g)`.
**Verification:** `uv run pytest tests/checkpoint/test_reuse_load.py -q` (skips if weights absent)
**Execution:** subagent recommended
**Depends on:** Slice 2, Slice 3
**Touches:** `src/mlx_speech/models/reuse/{semamba.py,block.py,loader.py}`, `scripts/convert/reuse.py`, `tests/checkpoint/test_reuse_load.py`, `tests/unit/test_reuse_semamba.py`

**Status:** complete
**Evidence:** Pure-MLX SEMamba (`semamba.py`, `mamba/block.py`) + conversion (`loader.py`, `scripts/convert/reuse.py`); all 1416 keys map exactly (strict `assert_keys_match`); forward returns correct shapes; no torch; ruff clean. Spec review APPROVED (line-by-line fidelity vs generator/mamba_block/codec references). Quality review caught a real backward-branch bug (scan-only reversal, conv not reversed) — FIXED so the whole backward module runs on flipped input (`flip(bwd(flip(x)) + flip(x))`), pinned by a unit test asserting it differs from the naive version (>1e-3). Quality follow-ups applied: assembly unit tests (`test_reuse_semamba.py`), removed dead `selective_scan_reverse`, deduped key-match. Verified: 506 unit + checkpoint passed.
**Risks / next:** Slice 6 parity must confirm InstanceNorm eps (config `norm_epsilon=1e-5` matches MLX default), conv causal-trim, and the atan2 phase branch.

### Slice 5: REUSEEnhancer wrapper + Mac self-consistency check

**Objective:** Assemble the chunked enhance() pipeline (STFT → SEMamba →
sweep filter → iSTFT → OLA) as a pure-MLX `REUSEEnhancer.from_dir(...).enhance(wav, in_sr)`.
**Acceptance criteria:**
- `enhance` returns a finite, length-matched waveform; no torch import on the path.
- On a synthetic noisy tone, output SNR improves vs input (self-consistency, runs on Mac).
**Verification:** `uv run pytest tests/runtime/test_reuse_enhance.py -q` (skips if weights absent)
**Depends on:** Slice 4
**Touches:** `src/mlx_speech/generation/reuse.py`, `tests/runtime/test_reuse_enhance.py`

### Slice 6: Torch parity gate (fixtures)

**Objective:** Add `scripts/eval/reuse_capture_reference.py` (torch RE-USE,
kernel-free path), capture fixtures, commit them, and gate the MLX enhancer
against them. This is the only end-to-end numeric proof, so it is a HARD
completion gate (AC3).
**Acceptance criteria:**
- Capture script writes `tests/fixtures/reuse/*.npz`, which are committed to the repo.
- MLX enhancer output vs fixtures: correlation ≥ 0.99 and bounded max-abs-diff.
- **Hard gate:** `test_reuse_parity.py` *errors* (not skips) if the committed fixtures are absent. The change cannot reach `verified` with this test skipped; `auto-verify` must confirm it ran green, not xfail/skip.
**Verification:** `uv run pytest tests/runtime/test_reuse_parity.py -q` runs green against the committed fixtures (a skipped result fails the gate).
**Depends on:** Slice 5
**Checkpoint after:** human-action
**Checkpoint reason:** fixture *capture* may require a host with `torch` + `mamba_ssm` (no macOS wheels); pause to capture there if not generatable locally. Once fixtures are committed, the gate runs anywhere.
**Touches:** `scripts/eval/reuse_capture_reference.py`, `tests/fixtures/reuse/`, `tests/runtime/test_reuse_parity.py`

### Slice 7: DramaBox integration (denoise_ref=True)

**Objective:** Wire `REUSEEnhancer` into `generate()`: clean the reference before
`waveform_to_mel` when `denoise_ref=True`, cache per clip, keep default `False`,
clear error when unavailable; remove the `NotImplementedError`. Collapse the
reference to mono before RE-USE and re-expand to the expected shape after,
mirroring `.references/DramaBox/src/inference_server.py:283-304`. Confine the
change to transforming `ref_audio.waveform` before `dramabox.py:279`; leave
`apply_reference_latent` and the denoise mask untouched.
**Acceptance criteria:**
- `generate(voice_ref, denoise_ref=True)` returns finite 48 kHz stereo, differs from `denoise_ref=False`, and caches the cleaned ref per `(path, sr)`.
- `denoise_ref=False` output is unchanged vs today (regression guard).
- `denoise_ref=True` with module/weights absent raises a clear error naming RE-USE and the `denoise_ref=False` opt-out.
- No torch import on the `denoise_ref=True` runtime path (asserted, not assumed).
**Verification:** `uv run pytest tests/unit/ tests/runtime/test_dramabox_reuse.py tests/runtime/test_reuse_purity.py -q` (purity test asserts `torch` is absent from `sys.modules` after the `denoise_ref=True` path runs).
**Execution:** subagent recommended
**Depends on:** Slice 6
**Touches:** `src/mlx_speech/generation/dramabox.py`, `src/mlx_speech/_hub.py`, `src/mlx_speech/tts/_adapters/dramabox.py`, `tests/unit/test_dramabox_reuse.py`, `tests/runtime/test_dramabox_reuse.py`, `tests/runtime/test_reuse_purity.py`

### Slice 8: Publish weights + docs

**Objective:** Publish the converted MLX RE-USE weights to `appautomaton` with the
NSCLv1 license + NVIDIA attribution + model card; update docs and plan.
**Acceptance criteria:**
- HF repo hosts the weights with `LICENSE` (NSCLv1), NVIDIA attribution, and a non-commercial model card.
- `resolve_reuse_path` defaults to the published repo; `docs/dramabox.md` + README document `denoise_ref` (opt-in, license note); plan marked done.
- `uv run pytest tests/unit/` green.
**Verification:** `hf auth whoami` + repo listing; `uv run pytest tests/unit/ -q`
**Depends on:** Slice 7
**Checkpoint after:** human-action
**Checkpoint reason:** the HF publish is outward-facing; confirm before pushing.
**Touches:** `scripts/hugging_face/upload.py`, model card, `docs/dramabox.md`, `README.md`, `plans/v5-dramabox.md`

## Aggregate verification

| Slice | Command |
| --- | --- |
| 1 | reference present + key/param count printed |
| 2 | `uv run pytest tests/unit/test_reuse_scan.py -q` |
| 3 | `uv run pytest tests/unit/test_reuse_stft.py -q` |
| 4 | `uv run pytest tests/checkpoint/test_reuse_load.py -q` |
| 5 | `uv run pytest tests/runtime/test_reuse_enhance.py -q` |
| 6 | `uv run pytest tests/runtime/test_reuse_parity.py -q` (hard gate: must run green, not skip) |
| 7 | `uv run pytest tests/unit/ tests/runtime/test_dramabox_reuse.py tests/runtime/test_reuse_purity.py -q` |
| 8 | `hf auth whoami`; `uv run pytest tests/unit/ -q` |

## Review: Engineering

**Reviewer:** Codex `gpt-5.5` (reasoning `xhigh`, `--sandbox read-only`, session `019ecb6b`). 2026-06-15.
**Verdict:** `needs_correction` (original) -> `approved_with_risks` (re-confirmed in the
same session). Both blockers and all four follow-ups verified resolved against the revised
plan. Watched risk: selective-scan numerics, gated by Slice 2's vendored `selective_scan_ref`
cases and Slice 6's fixture parity gate.

**Riskiest slice:** Slice 2 (selective scan) — one `dt`/A/B/C/D or bidirectional-combine
mismatch corrupts every enhanced chunk.

**Most likely failure mode:** Slice 2 matches its own numpy recurrence but not
`mamba_ssm.selective_scan_ref` (dt softplus/bias, A sign, variable B/C, D skip,
reverse combine), and Slice 4 only checks load + dummy shapes, so the error survives.

**Test-strategy gap (as originally written):** Slice 6 parity could be skipped when
fixtures are absent, and Slices 5/7 prove execution (finite/SNR/differs), not numerics.

### Blockers and resolution

1. **Slice 6 must be a hard gate** (was "skips if fixtures absent" → AC3 bypassable on
   this Mac). **Resolved:** Slice 6 now errors (not skips) without committed fixtures;
   `auto-verify` must confirm it ran green. Fixtures are committed.
2. **Slice 7 should gate on Slice 6, not Slice 5** (could integrate a self-consistent but
   numerically wrong enhancer). **Resolved:** Slice 7 `Depends on: Slice 6`; serial spine
   5 -> 6 -> 7 recorded in topology.

### Follow-ups and resolution

1. Slice 1 `python -c "..."` placeholder → **Resolved:** concrete `reuse_inspect.py` command.
2. Add scan cases for `dt_bias`/softplus, variable B/C, D skip, reverse/combine →
   **Resolved:** added to Slice 2 acceptance, tied to the vendored `selective_scan_ref`.
3. Add a no-`torch` runtime-purity guard → **Resolved:** `tests/runtime/test_reuse_purity.py`
   added to Slice 7 verification.
4. Specify mono collapse + re-expand (`inference_server.py:283-304`) → **Resolved:** added
   to Slice 7 objective.

**Blast radius:** Codex confirmed low risk to `denoise_ref=False` and Claims A/B provided
Slice 7 only transforms `ref_audio.waveform` before `dramabox.py:279`. Recorded in Slice 7.
