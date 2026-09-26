# DESIGN: DramaBox voice-clone conditioning (core, pure-MLX)

Scope: the non-obvious part of the SPEC — how a reference clip becomes appended,
read-only DiT tokens. The mel/encode/prep pieces are mechanical; this file covers
the conditioning + attention-mask pattern so execution doesn't reinvent it.

## Data flow

```
ref.wav
  → reference_prep: load → resample 16k → stereo → crop/loop 10s → peak -4 dBFS   (Slice 3)
  → AudioProcessor.waveform_to_mel  → mel [B,2,T_mel,64]                            (Slice 1)
  → AudioVAE.encode                 → ref_latent [B,8,T_lat,16] (normalized)        (Slice 2)
  → apply_reference_latent(state, ref_latent)                                       (Slice 5)
       · patchify ref_latent → ref_tokens [B, M, 128]
       · APPEND to end:  latent      = concat(target[B,N,128], ref_tokens)  → [B,N+M,128]
                         clean_latent = concat(target_clean,  ref_tokens)   → frozen copy
       · denoise_mask: target=1, ref=0   → [B, N+M, 1]   (drives TWO things: (a) post_process_latent
         re-blend in the loop AND (b) per-token AdaLN timestep = denoise_mask*sigma — see below)
       · positions:  ref = target_positions + 0.5s offset, concat → [B,1,N+M,2]
       · attention_mask: asymmetric [B,1,N+M,N+M] additive log-bias (see below)
  → euler_denoising_loop (mask + positions + per-token timesteps threaded into DiT)  (Slice 4 plumbing)
  → clear_conditioning: slice off trailing M tokens, reset mask                     (exists; used Slice 6)
  → unpatchify → silence_prior_fix → VAE.decode → vocoder
```

## Asymmetric attention mask

Token order is `[ target(0..N-1) | ref(N..N+M-1) ]`. Allowed (0.0) vs blocked
(large negative, additive pre-softmax):

| query \ key | target | ref |
|---|---|---|
| **target** | allow | **allow** (read identity) |
| **ref** | block | allow |

So target attends to ref (reads timbre), ref is read-only (cannot attend back to
target → its content can't be rewritten by the in-progress generation). Build a
`{0,1}` float matrix then convert to additive bias: allowed → `0.0`, blocked →
`mx.finfo(dtype).min` (matches upstream `_prepare_self_attention_mask`,
`transformer_args.py:97-123`). Shape `[B, 1, N+M, N+M]` to broadcast over heads.
RoPE positions for both segments come from one `precompute_split_freqs_from_positions`
call over the concatenated `[B,1,N+M,2]` — no rope change needed.

## Threading points (Slice 4)

TWO per-token signals must reach the DiT for the appended ref tokens. Threading
only the mask is **not** sufficient (that was the v1 defect — see below).

**(1) Self-attention mask.** The `mask` arg already reaches
`mx.fast.scaled_dot_product_attention` via `LTXAttention(..., mask=...)`. Missing links:
- `LTXBlock.__call__`: add `self_attention_mask=None`; pass to
  `self.audio_attn1(h, rope_cos_sin=..., mask=self_attention_mask)`.
- `LTXModel.__call__`: add `attention_mask=None`; forward to every block.
- `X0Model` + `euler_denoising_loop`: forward `state.attention_mask` into `x0_model(...)`.

**(2) Per-token timesteps — `timesteps = denoise_mask * sigma`.** Upstream the
per-step `Modality` carries per-token timesteps, NOT a scalar sigma
(`timesteps_from_mask` + `modality_from_latent_state`, `ltx_pipelines/utils/helpers.py:257-288`).
They drive the AdaLN modulation: `_prepare_timestep` (`transformer/transformer_args.py:62-75`)
embeds per-token → `[B, T, coeff*hidden]`, and `X0Model` denoises per-token
(`transformer/model.py:461-485`). Ref tokens (denoise_mask=0) → timestep **0** → AdaLN
modulates them as **clean/final**; target tokens → timestep sigma. With a scalar sigma
the ref tokens are modulated as fully noisy, corrupting their identity → audible
artifacts/echoing. Missing links:
- `dit/timestep.py` (`AdaLayerNormSingle` / `sinusoidal_timestep_embedding`): accept
  `[B, T]` timesteps, return per-token `[B, T, coeff*hidden]`.
- `dit/model.py` (`LTXModel.__call__`): compute `timesteps = denoise_mask * sigma` from a
  passed-in `denoise_mask`; feed per-token `ada_emb` to blocks and per-token
  `embedded_t` to the final-AdaLN output head (`:161-172`).
- `dit/block.py` (`LTXBlock.__call__`): consume per-token `[B, T, 9, dim]` factors (today
  `:115` reshapes to `[B, 1, 9, dim]`, i.e. one modulation broadcast to all tokens).
- `X0Model` + `euler_denoising_loop` + `generate`: pass `state.denoise_mask` alongside
  `positions`/`attention_mask`; `to_denoised` uses `denoise_mask * sigma`.

**Invariant:** when `denoise_mask` is all-ones (no voice ref) per-token reduces to the
current scalar path — the no-ref output **must stay bit-stable** after this refactor.

`context_mask` (cross-attn) is unrelated and stays as-is.

## Parity fixtures (verification)

- **Mel (Slice 1):** capture `torchaudio.transforms.MelSpectrogram` (documented params)
  on a fixed test waveform in a throwaway torch venv → commit small `.npy` fixture.
  No DramaBox weights needed.
- **Encode (Slice 2):** capture upstream audio-VAE `encode` output on a fixed mel using
  the vendored torch code + our local VAE weights → commit small `.npy` latent fixture.
- Fixtures are tiny derived arrays, not model weights — OK to commit under `tests/fixtures/`.
- **Fallback (SPEC risk R4):** if a torch venv can't be stood up, downgrade the two
  parity gates to self-consistency checks (mel framing invariants; encode→decode
  round-trip) and record the parity gate as an assumption — do not silently drop it.
