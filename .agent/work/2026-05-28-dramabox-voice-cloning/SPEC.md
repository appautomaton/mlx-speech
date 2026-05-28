# SPEC: DramaBox voice cloning (pure-MLX core)

## Bounded goal

Add transcript-free voice cloning to the pure-MLX DramaBox runtime: `DramaBoxModel.generate(prompt, voice_ref=<audio>)` conditions output timbre on a ~10 s reference clip, with no reference transcript and no torch in the runtime.

## Broader intent

Full upstream voice-clone parity. This SPEC is an intentional decomposition: it delivers the **core clone path** (reference used raw, `denoise_ref=False`). The optional RE-USE reference-denoise that upstream applies before encoding is deferred to the roadmap as optional quality-only — it's separable and not on the critical path (see Scope coverage).

## Work scale and shape

- **Scale:** medium capability, ~5 slices.
- **Shape:** feature + **parity** (numerical parity gate on the reference-encode stages against the vendored torch reference; perceptual confirmation on the full clone).

## Selected lenses

`engineering` + `runtime`. (Stakeholder is the downstream library user who wants to clone a voice; no UI/design surface.)

## Required outcome

New **input-side branch** `reference audio → mel → AudioVAE.encode → reference latent → appended read-only DiT tokens`, wired into the existing text→waveform path:

1. **Mel front-end** — implement `AudioProcessor.waveform_to_mel` (currently a stub) in pure MLX. Params are already correct in our config and must match upstream `ltx_core/.../audio_vae/ops.py`: 16 kHz, `n_fft=1024`, `win_length=1024`, `hop_length=160`, Hann, `center=True`, reflect pad, `power=1.0`, Slaney mel + Slaney norm, `n_mels=64`, `log(clamp(·,1e-5))`, output `[B,C,T_mel,64]`.
2. **VAE encode** — validate the *already-implemented* `AudioVAE.encode` / `AudioEncoder` (deterministic mean = channels 0..7, per-channel-stats normalize). Implementation exists; this is a validation task, not a build.
3. **Reference prep** — load WAV, resample to 16 kHz, force stereo, crop/loop to `ref_duration` (default 10 s), peak-normalize to **−4 dBFS**.
4. **Reference conditioning** (`AudioConditionByReferenceLatent` equivalent) — patchify the reference latent, compute its positions (target positions + 0.5 s offset), **append** ref tokens to the end of the latent sequence, set their `denoise_mask=0` (kept clean), and build an **asymmetric** self-attention bias `[B,1,N+M,N+M]` (target→ref allowed = 0.0; ref→target blocked = log-space `-inf`).
5. **Attention plumbing** — thread an `attention_mask` from `LTXModel.__call__` → `LTXBlock.__call__` → `audio_attn1` (the `mask` arg is already accepted by `LTXAttention` and forwarded to SDPA; only the two intermediate signatures need it). Concat positions are handled by the existing `precompute_split_freqs_from_positions` in one call.
6. **API + cleanup** — add `voice_ref` (and `denoise_ref=False` placeholder) to `DramaBoxModel.generate`; `clear_conditioning` strips the M appended ref tokens before VAE decode so output length is unchanged.

## Acceptance criteria

1. **[parity]** `waveform_to_mel` matches torchaudio `MelSpectrogram` on a fixed test signal within tolerance (log-mel max-abs diff threshold set in plan; starting target ≤ 1e-2).
2. **[parity]** `AudioVAE.encode` on a fixed 10 s reference matches upstream `vae_encode_audio` (`denoise_ref=False`) latent within tolerance (starting target: per-channel cosine ≥ 0.99).
3. **[unit]** Reference prep yields stereo, exactly `ref_duration·16000` samples, peak −4 dBFS (±0.1 dB) from a mono test clip.
4. **[unit]** Conditioning builds M ref tokens, `denoise_mask=0` on them, ref positions offset +0.5 s, and an asymmetric `[B,1,N+M,N+M]` bias with the correct allowed/blocked pattern.
5. **[runtime]** DiT runs with appended ref tokens + mask with no shape error; `clear_conditioning` removes exactly M tokens; decoded output length equals the no-ref run.
6. **[smoke]** `generate(prompt, voice_ref=clip)` returns finite, clamped `[-1,1]`, correct-length 48 kHz stereo, and is materially different from the no-ref output (conditioning has effect).
7. **[perceptual]** Cloned voice audibly tracks the reference across 2–3 prompts (A/B vs no-ref), previewed into `outputs/`.
8. `pytest tests/unit/` green; new unit + checkpoint tests for slices 1–5.

## Constraints and risks

**Constraints**
- Pure-MLX runtime — no torch in inference. Parity **fixtures** may be captured offline in a torch env, but runtime stays MLX.
- `denoise_ref=False` for this change (RE-USE deferred). Local-path-first; no `huggingface_hub` at runtime. No reference transcript.
- Reference input limited to soundfile-readable formats (WAV/FLAC); arbitrary containers deferred.

**Risks (implementation-changing)**
- **R1 — mel parity (MEDIUM, top risk):** matching torchaudio `center=True` reflect framing + Slaney filterbank norm in pure MLX without float drift; drives criterion 1/2.
- **R2 — mask correctness (MEDIUM):** T-dimension boundary off-by-one on the N+M concat and log-bias value (`-inf` vs `finfo.min`) per upstream `_prepare_self_attention_mask`.
- **R3 — causal Conv2d (EASY–MED):** validate height/time-axis causal padding in the existing encoder against a captured reference latent.
- **R4 — parity harness (setup):** numerical gates 1–2 require standing up the vendored torch DramaBox env once to capture fixtures with `denoise_ref=False`. If that env can't be stood up, fall back to perceptual-only for those stages and downgrade gates 1–2 to assumptions (recorded, not silently dropped).
- **R5 — resampler (LOW):** need a pure-MLX/numpy resampler for non-16 kHz reference WAVs.

## Scope coverage

**Included:** mel front-end, VAE-encode validation, reference prep, reference-latent conditioning + asymmetric mask, attention-mask plumbing, `voice_ref` API + `clear_conditioning`, parity + perceptual verification.

**Deferred (→ ROADMAP, `status: pending`):** RE-USE reference-denoise — optional pre-encode speech enhancer (SEMamba, ~9.6M-param Mamba/SSM) matching upstream `denoise_ref=True`. Deferred because it's optional quality-only (core clone runs `denoise_ref=False` with graceful fallback) and a separate self-contained model — not because it's infeasible. Porting needs a hand-written selective-scan (no prebuilt MLX kernel) and the `nvidia/RE-USE` weights fetched + converted (check license before redistributing).

**Anti-goals:** no reference transcript path; no torch in the MLX runtime; no RE-USE/SEMamba in this change; no multi-reference / speaker-blend; no other TTS feature work.

## Assumptions

- Parity tolerance thresholds are starting targets; finalized in the plan against captured fixtures.
- "Perceptual match" is a human listen check, not an automated speaker-similarity metric (no new model added for scoring).
