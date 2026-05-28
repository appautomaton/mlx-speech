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
       · denoise_mask: target=1, ref=0   → [B, N+M, 1]   (ref kept clean via post_process_latent)
       · positions:  ref = target_positions + 0.5s offset, concat → [B,1,N+M,2]
       · attention_mask: asymmetric [B,1,N+M,N+M] additive log-bias (see below)
  → euler_denoising_loop (mask + positions threaded into DiT)                       (Slice 4 plumbing)
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

The `mask` arg already reaches `mx.fast.scaled_dot_product_attention` via
`LTXAttention(..., mask=...)`. Missing links, all additive:
- `LTXBlock.__call__`: add `self_attention_mask=None`; pass to
  `self.audio_attn1(h, rope_cos_sin=..., mask=self_attention_mask)` (today line 120 has no mask).
- `LTXModel.__call__`: add `attention_mask=None`; forward to every block.
- `X0Model` + `euler_denoising_loop`: forward optional `attention_mask` from
  `state.attention_mask` into `x0_model(...)` alongside the existing `positions`.

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
