# SPEC: RE-USE voice-reference denoising (pure-MLX)

## Bounded goal

Port NVIDIA RE-USE (SEMamba speech enhancement) to a pure-MLX runtime and wire
it into DramaBox so `denoise_ref=True` cleans the input voice reference before
VAE conditioning, with no torch in the runtime path.

## Broader intent

Close the last unimplemented DramaBox item (`denoise_ref`, currently raises
`NotImplementedError` at `src/mlx_speech/generation/dramabox.py:242`) so a noisy
reference recording produces a cleaner voice clone. Voice cloning already works
without it (IC-LoRA conditioning + per-token sigma are shipped and verified), so
this is an opt-in quality refinement, not the cloning mechanism.

## Work scale and shape

- **Scale:** Capability-sized. New model port (architecture + weight conversion
  + numerical parity + integration). Smaller than a full TTS family (~9.6M
  params) but the same shape of work.
- **Shape:** Parity port (match the torch reference numerically) + integration
  into the existing DramaBox voice-ref path.

## Selected lenses

`engineering` (primary: SSM port + parity), `product` (exposure, default,
licensing posture), `runtime` (memory/latency on Apple Silicon; minor given
size, but selective-scan throughput matters).

## Target stakeholder

mlx-speech users on Apple Silicon doing DramaBox voice cloning from imperfect
reference recordings. Secondarily the maintainer, for OSS positioning around a
non-commercial-licensed dependency.

## Constraints (that change implementation)

- **Pure MLX runtime, no torch** on the `denoise_ref=True` path (CLAUDE.md hard
  rule). Torch is allowed only for offline parity capture / conversion.
- **Core port = the selective-scan (SSM recurrence).** STFT/iSTFT (mag+phase),
  causal conv1d, linear, and gating map straightforwardly to MLX; the bidirectional
  selective scan is the one non-trivial primitive.
- **Reference behavior to match** (`.references/DramaBox/src/super_resolution.py`):
  STFT mag/phase with `compress_factor`; per-chunk model forward; the "sweep
  artifact" filter (`expm1(relu(amp))`, zero-portion masking); Hann overlap-add
  at 50% hop; denoise-only with `target_sr == in_sr` (no resample, no BWE).
- **Weights:** `nvidia/RE-USE`, license **NSCLv1 (non-commercial)**. Converted
  MLX weights hosted on the `appautomaton` HF org (user decision). That repo MUST
  carry the NSCLv1 license + NVIDIA attribution + a non-commercial model card.
  Library code stays MIT. `.safetensors`; weights never in git.
- **Reuse existing plumbing.** RE-USE inserts before `AudioProcessor.waveform_to_mel`
  in the voice-ref path; no changes to `apply_reference_latent` (Claim A) or
  per-token sigma (Claim B). Cache the cleaned reference per clip (mirror
  `_denoise_voice_ref`).
- **Default `denoise_ref=False`** (user decision): the default voice-clone path
  stays pure-MLX with zero non-commercial dependency.

## Risks

- **Selective-scan parity** is the primary risk. Mitigate with a primitive-level
  gate (MLX scan vs a numpy/torch-CPU reference recurrence) before full-model
  parity.
- **Parity capture environment:** `mamba_ssm` has no macOS wheels, so capturing
  the full torch reference on this Mac may not be possible. Mitigate by (a)
  capturing reference fixtures once on a Linux/CPU box and comparing MLX against
  saved arrays, and/or (b) validating the scan primitive against a self-written
  numpy reference and the full model against saved fixtures. Resolve in plan.
- **Bidirectional composition** (forward+reverse scan) must match SEMamba.
- **STFT details** (`compress_factor`, sweep-artifact filter, OLA window/hop)
  must match exactly or output differs audibly.
- **Weight key mapping** SEMamba -> MLX modules.

## Required outcome

- **Behavior:** `DramaBoxModel.generate(..., voice_ref=path, denoise_ref=True)`
  runs end-to-end in pure MLX, returning finite 48 kHz stereo, with the reference
  cleaned by MLX RE-USE before VAE encoding. The cleaned reference is cached per
  clip.
- **Standalone module:** an MLX SEMamba that loads converted weights and exposes
  `clean = reuse(noisy_waveform, in_sr)`.
- **Parity target:** MLX RE-USE enhanced output matches the torch RE-USE
  reference (kernel-free path) on a fixed noisy clip, within a tolerance set in
  the plan (e.g. waveform correlation >= 0.99 and bounded max-abs-diff).

## Acceptance criteria

1. **Unit:** MLX selective-scan matches a numpy/torch-CPU reference recurrence on
   random inputs within tolerance (the core primitive gate).
2. **Checkpoint:** converted SEMamba weights load into the MLX module with every
   key mapped (no missing/extra keys).
3. **Parity (runtime, fixtures or torch venv):** MLX enhanced output vs the torch
   reference on a fixed clip meets the correlation / max-abs-diff thresholds.
4. **Integration (runtime):** `generate(voice_ref, denoise_ref=True)` returns
   finite 48 kHz stereo and differs from `denoise_ref=False`; reference cleaning
   is cached per clip.
5. **Regression:** `denoise_ref=False` output is unchanged vs today.
6. **Pure MLX:** no torch import on the `denoise_ref=True` runtime path.
7. `pytest tests/unit/` is green.
8. The `appautomaton` RE-USE repo carries the NSCLv1 license + NVIDIA attribution.

## Anti-goals

- No bandwidth extension / super-resolution; denoise-only.
- No torch in the runtime path.
- No changes to IC-LoRA conditioning (Claim A) or per-token sigma (Claim B).
- Do not flip the `denoise_ref` default to True.
- Not a general speech-enhancement framework; SEMamba/RE-USE only.
- No output-side denoising (would suppress generated paralinguistics).

## Scope coverage

- **Included:** SEMamba MLX port, weight conversion, selective-scan primitive +
  parity gate, DramaBox `denoise_ref=True` integration with per-clip cache,
  appautomaton hosting with NSCLv1 license, docs update.
- **Deferred / not in scope:** BWE, output denoising, alternative enhancers,
  flipping the default, A/B changes.
- **Decided this conversation:** default `False`; host converted weights on
  `appautomaton` (carrying NSCLv1); denoise-only.

## Assumptions

- When `denoise_ref=True` but the RE-USE module/weights are unavailable, raise a
  clear error naming the cause (opt-in implies the user wants it). Upstream warns
  and skips; revisit in plan if a softer fallback is preferred.
- Parity is captured against the kernel-free torch path; no CUDA required.
