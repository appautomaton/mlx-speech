# PLAN: DramaBox voice cloning (pure-MLX core)

## Goal

Transcript-free voice cloning in the MLX runtime: `DramaBoxModel.generate(prompt, voice_ref=<wav>)` conditions output timbre on a ~10 s reference clip (`denoise_ref=False`). Full contract: `SPEC.md`. Design for the conditioning/mask pattern: `DESIGN.md`.

## Architecture approach

New input-side branch `ref audio → mel → AudioVAE.encode → appended read-only DiT tokens`, gated by an asymmetric self-attention mask. Most scaffolding already exists (`LatentState.attention_mask`, `clear_conditioning`, `LTXAttention(mask=...)`, `AudioVAE.encode`). See `DESIGN.md` for token-append order, mask layout, and threading points.

## Ordered slice sequence

### Slice 1: Mel front-end

Required:
**Objective:** Implement `AudioProcessor.waveform_to_mel` in pure MLX (currently a stub) matching upstream `ops.py` params.
**Acceptance criteria:** (SPEC AC1)
- `waveform_to_mel` returns `[B, C, T_mel, 64]` log-mel for a stereo input.
- Output matches a committed torchaudio `MelSpectrogram` fixture within tolerance (start target: log-mel max-abs ≤ 1e-2).
**Verification:** `.venv/bin/python -m pytest tests/unit/test_dramabox_audio_processor.py -q`
**Touches:** `src/mlx_speech/models/dramabox/audio_vae/audio_processor.py`, `tests/unit/test_dramabox_audio_processor.py`, `tests/fixtures/`
**Produces:** working mel front-end + parity fixture.
**Status:** complete
**Evidence:** changed `src/mlx_speech/models/dramabox/audio_vae/audio_processor.py`, added `tests/unit/test_dramabox_audio_processor.py`, and generated `tests/fixtures/dramabox/audio_processor_mel_fixture.npz`; `.venv/bin/python -m pytest tests/unit/test_dramabox_audio_processor.py -q` passed (`3 passed`); `.venv/bin/python -m pytest tests/unit/` passed (`300 passed`).
**Risks / next:** none.

### Slice 2: VAE-encode validation

Required:
**Objective:** Exercise and validate the existing `AudioVAE.encode` (means-only + per-channel-stats normalize) against an upstream reference latent.
**Acceptance criteria:** (SPEC AC2)
- `encode(mel)` returns normalized latent `[B, 8, T_lat, 16]`, finite.
- Matches a committed upstream encode fixture within tolerance (start target: per-channel cosine ≥ 0.99). Fallback per `DESIGN.md`/R4 if no torch env: encode→decode round-trip + distribution check, recorded as assumption.
**Verification:** `.venv/bin/python -m pytest tests/checkpoint/test_dramabox_audio_vae_checkpoint.py -q -k encode`
**Execution:** subagent recommended
**Touches:** `src/mlx_speech/models/dramabox/audio_vae/{model.py,encoder_decoder.py}`, `tests/checkpoint/test_dramabox_audio_vae_checkpoint.py`, `tests/fixtures/`
**Status:** complete
**Evidence:** generated `tests/fixtures/dramabox/audio_vae_encode_fixture.npz` from `.venv-torch` upstream AudioEncoder on a fixed 10 s stereo mel and added encode parity coverage in `tests/checkpoint/test_dramabox_audio_vae_checkpoint.py`; `.venv/bin/python -m pytest tests/checkpoint/test_dramabox_audio_vae_checkpoint.py -q -k encode` passed (`1 passed, 2 deselected`); `.venv/bin/python -m pytest tests/unit/` passed (`300 passed`).
**Risks / next:** none.

### Slice 3: Reference prep

Required:
**Objective:** Loader that turns a reference file into the encoder's expected waveform.
**Acceptance criteria:** (SPEC AC3)
- From a mono test clip: output is stereo, exactly `ref_duration*16000` samples, peak −4 dBFS (±0.1 dB); shorter inputs loop, longer inputs crop.
**Verification:** `.venv/bin/python -m pytest tests/unit/test_dramabox_reference_prep.py -q`
**Touches:** new `src/mlx_speech/models/dramabox/audio_vae/reference_prep.py` (or `generation/`), `tests/unit/test_dramabox_reference_prep.py`
**Produces:** pure-MLX/numpy resample + normalize helper.
**Status:** complete
**Evidence:** added `src/mlx_speech/models/dramabox/audio_vae/reference_prep.py` and `tests/unit/test_dramabox_reference_prep.py`; `.venv/bin/python -m pytest tests/unit/test_dramabox_reference_prep.py -q` passed (`3 passed`); `.venv/bin/python -m pytest tests/unit/` passed (`303 passed`).
**Risks / next:** none.

### Slice 4: Self-attention mask plumbing

Required:
**Objective:** Thread an optional self-attention `attention_mask` through `LTXModel` → `LTXBlock` → `audio_attn1`, and through `X0Model` + `euler_denoising_loop`.
**Acceptance criteria:** (SPEC AC5, part)
- An all-allow (zeros) mask yields output identical to no-mask (within fp tolerance).
- A mask blocking a token subset measurably changes output vs no-mask.
- Existing DiT/sampling tests still pass.
**Verification:** `.venv/bin/python -m pytest tests/unit/test_dramabox_dit.py tests/unit/test_dramabox_sampling.py -q -k "mask or sampling"`
**Execution:** subagent recommended
**Touches:** `dit/model.py`, `dit/block.py`, `sampling/x0_model.py`, `sampling/loop.py`, `tests/unit/test_dramabox_dit.py`

### Slice 5: Reference-latent conditioning

Required:
**Objective:** `apply_reference_latent(state, ref_latent, ...)` — append ref tokens (positions +0.5 s, `denoise_mask=0`) and build the asymmetric `[B,1,N+M,N+M]` log-bias mask per `DESIGN.md`.
**Acceptance criteria:** (SPEC AC4, AC5)
- Appends M ref tokens to `latent`/`clean_latent`; `denoise_mask` is 1 for target, 0 for ref; positions extended with +0.5 s offset.
- `attention_mask` matches the allowed/blocked pattern (target→ref allow, ref→target block).
- `AudioLatentTools.clear_conditioning` restores target-only state (exactly N tokens, mask reset).
**Verification:** `.venv/bin/python -m pytest tests/unit/test_dramabox_conditioning.py -q`
**Depends on:** Slice 4
**Execution:** subagent recommended
**Touches:** new `src/mlx_speech/models/dramabox/diffusion/conditioning.py`, `tests/unit/test_dramabox_conditioning.py`

### Slice 6: API wiring + end-to-end

Required:
**Objective:** Add `voice_ref` (+ `denoise_ref=False` placeholder) to `DramaBoxModel.generate`; run prep→mel→encode→`apply_reference_latent`→loop(with mask)→`clear_conditioning`→decode; preview into `outputs/`.
**Acceptance criteria:** (SPEC AC6, AC7)
- `generate(prompt, voice_ref=clip)` returns finite, clamped `[-1,1]`, correct-length 48 kHz stereo; decoded length equals the no-ref run (ref tokens stripped).
- Output materially differs from the no-ref run (conditioning has effect).
- Perceptual: cloned voice tracks the reference across 2–3 prompts (human listen), previews written to `outputs/dramabox/`.
**Verification:** `.venv/bin/python -m pytest tests/runtime/test_dramabox_smoke.py -q -k voice_ref` then `.venv/bin/python -m pytest tests/unit/ -q`
**Depends on:** Slice 1, Slice 2, Slice 3, Slice 5
**Execution:** subagent recommended
**Checkpoint after:** human-verify
**Checkpoint reason:** Perceptual voice-match (AC7) requires a human to listen to the previews — cannot be auto-asserted.
**Touches:** `src/mlx_speech/generation/dramabox.py`, `scripts/generate_dramabox.py`, `tests/runtime/test_dramabox_smoke.py`

## Execution routing and topology

- **Default:** continuation — after each slice's verification passes, proceed to the next approved slice. Execution windows are context batches, not stopping points.
- **Parallel-safe groups:** {Slice 1, Slice 2, Slice 3, Slice 4} — independent, disjoint write sets (mel front-end / VAE-encode / reference-prep / DiT-sampling). Slice 5 then Slice 6 are serial.
- **Subagent routes:** Slices 2, 4, 5, 6 `subagent recommended` (cross-subsystem / interface work). Slices 1, 3 `direct`.
- **Checkpoints:** only after Slice 6 (`human-verify`, perceptual listen). No other checkpoints.
- **Fixture dependency (Slices 1–2):** parity fixtures captured offline in a throwaway torch venv (not added to runtime deps); if unavailable, apply the R4 self-consistency fallback and record the gate as an assumption rather than dropping it.

## Requirement traceability

| SPEC acceptance | Slice |
|---|---|
| AC1 mel parity | 1 |
| AC2 encode parity | 2 |
| AC3 reference prep | 3 |
| AC4 conditioning tokens/mask | 5 |
| AC5 DiT runs w/ ref tokens + clear strips | 4 + 5 + 6 |
| AC6 end-to-end smoke | 6 |
| AC7 perceptual | 6 (human-verify) |
| AC8 unit green + new tests | all |

## Aggregate verification commands

| Slice | Command |
|---|---|
| 1 | `.venv/bin/python -m pytest tests/unit/test_dramabox_audio_processor.py -q` |
| 2 | `.venv/bin/python -m pytest tests/checkpoint/test_dramabox_audio_vae_checkpoint.py -q -k encode` |
| 3 | `.venv/bin/python -m pytest tests/unit/test_dramabox_reference_prep.py -q` |
| 4 | `.venv/bin/python -m pytest tests/unit/test_dramabox_dit.py tests/unit/test_dramabox_sampling.py -q -k "mask or sampling"` |
| 5 | `.venv/bin/python -m pytest tests/unit/test_dramabox_conditioning.py -q` |
| 6 | `.venv/bin/python -m pytest tests/runtime/test_dramabox_smoke.py -q -k voice_ref` + `.venv/bin/python -m pytest tests/unit/ -q` |
| gate | `.venv/bin/python -m pytest tests/unit/` green after every slice |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The slices fit the existing DramaBox boundaries and keep the reference branch local to audio prep, VAE encode, latent conditioning, DiT mask plumbing, and generate wiring.
- Concern: The riskiest path is Slices 4-5 because a mask or position threading miss can produce a shape-valid loop where target tokens do not reliably read reference identity, though the planned mask and runtime-effect tests should catch it.
- Action: In `auto-execute`, implement Slice 4 before Slice 5 and verify the conditioned loop passes `state.positions` and `state.attention_mask` together with all-allow, blocked-subset, clear-conditioning, and voice-ref smoke tests.
- Verified: Read PLAN.md, DESIGN.md, SPEC.md, skill references, and the DramaBox audio processor, VAE, latent state, patchifier, DiT block/model, X0 loop, generation wrapper, and current tests.
