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
**Status:** complete
**Evidence:** threaded additive self-attention masks through `src/mlx_speech/models/dramabox/dit/{model.py,block.py}` and `src/mlx_speech/models/dramabox/sampling/{x0_model.py,loop.py}`, with mask equivalence/effect tests in `tests/unit/test_dramabox_dit.py` and loop propagation coverage in `tests/unit/test_dramabox_sampling.py`; `.venv/bin/python -m pytest tests/unit/test_dramabox_dit.py tests/unit/test_dramabox_sampling.py -q -k "mask or sampling"` passed (`16 passed, 7 deselected`); `.venv/bin/python -m pytest tests/unit/` passed (`306 passed`).
**Risks / next:** superseded by VERIFY-GAP below.

> **VERIFY-GAP (2026-05-28, auto-verify) — per-token timestep never threaded through the DiT AdaLN. [RESOLVED 2026-05-28 gap-fix]**
> **Severity:** blocking (root cause of AC7 perceptual failure).
> **Resolution:** threaded an optional per-token `denoise_mask` through `dit/timestep.py` (`AdaLayerNormSingle`/`sinusoidal_timestep_embedding` accept `[B,T]`), `dit/model.py` (per-token `timesteps = denoise_mask*sigma` → AdaLN; prompt-AdaLN stays on scalar sigma; final head per-token-aware), `dit/block.py` (per-token `[B,T,9,dim]` factors), `sampling/{x0_model,loop}.py`, and `generation/dramabox.py` (passes `state.denoise_mask` only when `voice_ref`; `None` → bit-identical scalar no-ref path). `tests/unit/` → 317 passed; new coverage asserts no-ref `denoise_mask=None` equals the omitted/scalar path (atol 0), frozen tokens change output, and `X0Model` returns ref tokens (mask 0) unchanged. Empirical A/B (6 s, 30 steps, cfg 2.5, seed 42): no-ref byte-identical (rms 0.0577, flatness 0.072); voice-ref moved off the artifact signature (flatness 0.0225→0.0578, centroid 2663→1860 Hz toward ref 1682 Hz, peak 1.000 clipping→0.498). Final perceptual sign-off = Slice 6 human-verify.
> **Evidence (fresh, upstream + our code):**
> - Upstream builds each step's Modality with `timesteps = denoise_mask * sigma` (`.references/DramaBox/ltx2/ltx_pipelines/utils/helpers.py:271,279-288` `timesteps_from_mask`; `modality_from_latent_state` :267-276). These per-token timesteps drive AdaLN: `_prepare_timestep` (`.../transformer/transformer_args.py:62-75`) flattens per-token, embeds, and reshapes to `[B, T, coeff*hidden]`; the X0Model also denoises per-token (`.../transformer/model.py:461-485`, `to_denoised(latent, velocity, audio.timesteps)`). Reference tokens (`denoise_mask=0`) → timestep `0` → modulated as **clean**; target tokens → timestep `sigma`.
> - Our DiT embeds a **scalar per-batch** sigma only: `dit/model.py:141-143` (`sigma_scaled = sigma * 1000`; `audio_adaln_single(sigma_scaled)` → `ada_emb [B, coeff*hidden]`), `dit/timestep.py:90-98` (`AdaLayerNormSingle` takes `[B]`), `dit/block.py:115` (`ada_emb.reshape(B, 1, 9, dim)` → identical modulation broadcast to every token). `sampling/x0_model.py:60` uses scalar `float(sigma[0])`. Reference tokens are therefore modulated at the noisy timestep (≈1000 at step 0) instead of `0`.
> - Effect: corrupted reference latents inside the transformer; target tokens attend to garbled identity → audible artifacts + echoing on the voice-ref path (user-confirmed) while the no-ref path is clean (with `denoise_mask` all-ones, per-token ≡ per-batch). The mask/positions/append-order all match upstream — they are not the defect.
> **Fix objective:** thread per-token timesteps (`denoise_mask * sigma`) through the DiT so reference tokens are AdaLN-modulated as clean. Touch points: `dit/timestep.py` (`AdaLayerNormSingle`/`sinusoidal_timestep_embedding` accept `[B, T]` and return per-token `[B, T, coeff*hidden]`), `dit/model.py` (compute per-token timesteps from a passed-in `denoise_mask`, feed per-token `ada_emb` + per-token `embedded_t` to the final AdaLN at :161-172), `dit/block.py:115` (consume `[B, T, 9, dim]`), `sampling/x0_model.py` (`to_denoised` with `denoise_mask * sigma`), and `generation/dramabox.py` + `sampling/loop.py` (pass `state.denoise_mask` into the X0Model/DiT). Verify the all-ones (no-ref) path is bit-stable vs current, and add a unit test asserting ref-token modulation uses timestep 0 (e.g., a 2-token toy where masking a token changes only its AdaLN factors).

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
**Status:** complete
**Evidence:** added `src/mlx_speech/models/dramabox/diffusion/conditioning.py`, exported `apply_reference_latent`, and added `tests/unit/test_dramabox_conditioning.py`; `.venv/bin/python -m pytest tests/unit/test_dramabox_conditioning.py -q` passed (`3 passed`); `.venv/bin/python -m pytest tests/unit/` passed (`309 passed`).
**Risks / next:** none.

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
**Status:** complete
**Evidence:** wired `voice_ref` and `denoise_ref=False` through `src/mlx_speech/generation/dramabox.py`, added CLI flags in `scripts/generate_dramabox.py`, and added runtime A/B coverage in `tests/runtime/test_dramabox_smoke.py`; `.venv/bin/python -m pytest tests/runtime/test_dramabox_smoke.py -q -k voice_ref` passed (`1 passed, 1 deselected`); `.venv/bin/python -m pytest tests/unit/` passed (`309 passed`); generated finite A/B previews in `outputs/dramabox/prompt{1,2}_{no_ref,voice_ref}.wav`.
**Risks / next:** superseded by VERIFY-GAP below.

> **VERIFY-GAP (2026-05-28, auto-verify) — AC7 perceptual FAIL; previews are smoke-settings artifacts. [RESOLVED 2026-05-28 gap-fix]**
> **Severity:** blocking (AC7 not met → plan fails).
> **Resolution:** Slice-4 per-token-timestep root cause fixed; real-settings A/B previews regenerated into `outputs/dramabox/prompt{1,2}_{no_ref,voice_ref}.wav` (6 s, 30 steps, cfg 2.5; outputs/ is gitignored, local-only). **Human-verify checkpoint PASSED (2026-05-28):** user confirmed the cloned voice is clean and tracks the reference (artifacts/echoing gone). AC7 satisfied.
> **Evidence (fresh):**
> - The committed previews `outputs/dramabox/prompt{1,2}_{no_ref,voice_ref}.wav` were generated with **smoke-test parameters** (`duration_s=1.0, steps=3, cfg_scale=1.0`; file mtimes 20:28:02–20:28:14, 3–5 s apart): all 1.61 s, near-silent (rms 0.004–0.015, envmod 0.47). They are unfit for AC7 perceptual judgment — they are not real previews.
> - Fresh real-settings A/B (6 s, 30 steps, cfg 2.5, seed 42, identical prompt; only `voice_ref` differs): no-ref is clean, speech-like (rms 0.058, flatness 0.072, envmod 1.43, comparable to the known-good `dive_ep01`). voice-ref is finite/clamped but **perceptually bad — artifacts + echoing (human listen, 2026-05-28)**.
> - Root cause is the Slice-4 VERIFY-GAP (per-token timestep not threaded through the DiT AdaLN). AC6's "materially different" assertion passed but cannot distinguish working cloning from corruption, so it gave false confidence.
> **Fix objective:** (1) fix the Slice-4 per-token-timestep gap; (2) regenerate AC7 A/B previews across 2–3 prompts at real settings (≥5 s, 30 steps, cfg 2.5) into `outputs/dramabox/` and pass the human-verify checkpoint (cloned voice tracks the reference); (3) strengthen the smoke/runtime test beyond "materially different" — e.g. assert the no-ref path is unchanged by the per-token refactor and that ref tokens carry timestep 0. **Depends on:** Slice 4 fix.

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
