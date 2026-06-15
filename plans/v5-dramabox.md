# v5: DramaBox — MLX-Native Inference, Text → 48 kHz Waveform

**Status: BASELINE LANDED** (2026-05-27 — rev4 plan, end-to-end smoke test passes)

## Implementation status

| Stage | Status |
| --- | --- |
| 1 — Plan + repo registration | ✅ Done |
| 2a — Gemma 3 text backbone | ✅ Done (loads, all 49 hidden states forward; 24 tests) |
| 2b — Prompt pipeline | ✅ Done (FeatureExtractorV2 + Connector load + forward; 15 tests) |
| 3 — Latent grid, scheduler, noiser | ✅ Done (26 tests) |
| 4 — AudioVAE | ✅ Done (encoder + decoder + per-channel-stats; 102 keys load; 15 tests) |
| 5 — Vocoder stack | ✅ Done (main BigVGAN + BWE + mel-STFT; 1227 keys load; 9 tests) |
| 6 — LTX DiT | ✅ Done (48 layers, 1457 keys load; 8 tests) |
| 7 — Guidance + denoising loop | ✅ Done (CFG + rescale + STG; IC-LoRA denoise-ref deferred) |
| 8 — Wrapper + CLI + smoke test | ✅ Done (`DramaBoxModel.generate` runs end-to-end) |
| 9 — Docs + release readiness | ✅ Done (`docs/dramabox.md`, README, references) |

**Test sweep**: 297 unit tests pass; 8 checkpoint tests pass (Gemma 4-bit,
prompt pipeline, AudioVAE, vocoder, DiT); 1 runtime smoke test passes
(end-to-end generation produces a finite 48 kHz stereo waveform with
non-zero RMS).

## Landed since baseline

- **STG (Spatio-Temporal Guidance)** — Done. A `skip_self_attn` flag threads
  through `LTXAttention → LTXBlock → LTXModel → X0Model`, and the loop runs a
  third perturbed pass (positive context, value passthrough on `stg_blocks`,
  default block 29) that the guider folds in as `stg_scale * (cond - ptb)`.
  Default `stg_scale` flipped `0.0 → 1.5` to match the warm-server reference
  (`inference_server.py:307`). Verified by `tests/unit/test_dramabox_stg.py`
  and a runtime STG-vs-CFG-only A/B (`tests/runtime/test_dramabox_stg_runtime.py`).

## Known deferred items (follow-ups)

These do not block end-to-end audio output:

1. **Refined voice-reference conditioning (IC-LoRA, `denoise_ref=True`)** —
   The raw voice-reference path (`denoise_ref=False`) works:
   `AudioProcessor.waveform_to_mel` is implemented (STFT + mel filter bank)
   and the reference latent is appended with a per-token denoise mask. The
   denoised path raises `NotImplementedError`. To enable: wire
   `AudioConditionByReferenceLatent` (asymmetric attention mask) for the
   denoise-ref case.
2. **Per-token sigma for IC-LoRA** — Broadcast-per-batch sigma is correct on
   the no-ref path; the raw-ref path already uses per-token timesteps. Only
   the IC-LoRA denoise path above needs further per-token sigma work.

## Summary


Bring Resemble DramaBox into `mlx-speech` as a new TTS family with full
end-to-end waveform output, inference only, pure-MLX runtime.

Input is text (optionally a 10 s voice reference WAV). Output is a 48 kHz
stereo WAV that matches the DramaBox warm-server reference closely enough that
the port is trustworthy at the standard "proper settings"
(`cfg=2.5 stg=1.5 stg_block=29 rescale=auto modality=1.0 steps=30 fps=25
seed=42`, dev checkpoint).

DramaBox is a flow-matching diffusion audio model:

- text path: Gemma 3 12B IT → all 49 hidden states → `FeatureExtractorV2`
  (per-token RMS norm + rescale) → `audio_aggregate_embed`
  ([2048, 188160] linear at bf16, 770 MB; loaded plainly per the reference)
  → `audio_embeddings_connector` (an **8-block** 1D self-attention transformer
  over the full 1024-token text sequence; 128 learnable registers **replace
  padded slots** in that sequence via tile-and-mix; per-head gated attention;
  output `a_ctx [B, 1024, 2048]` with attention_mask = 0 everywhere because
  every slot is now valid)
- LTX-2.3-derived **audio-only DiT (3.3B params)**: 48 layers, 32 heads ×
  head_dim=64, **split RoPE on one temporal axis** with NumPy-float64 grid
  construction (cast to float32 before sin/cos), per-head **gated attention**,
  AdaLN modulation on both self- and cross-attention, **returns velocity**
- X0Model wrapper converts velocity → `x0` via `to_denoised(noisy, v, timesteps)`
- audio VAE — mel-spectrogram autoencoder with z_channels=8, 16 latent
  mel-bins, 2-channel stereo, 16 kHz input rate, downsample factor 4;
  has a separate `AudioProcessor` mel front-end (resample + MelSpectrogram +
  log clamp) on the encoder side
- vocoder stack — main BigVGAN (mel → 16 kHz) + BWE BigVGAN (16 → 48 kHz),
  forced to **fp32 compute** because bf16 degrades spectral metrics 40-90%

The novel feature versus existing families is **IC-LoRA voice cloning** — a
reference audio latent is patchified, appended to the end of the target token
sequence, and gated by an asymmetric attention mask so target tokens read the
reference but reference tokens stay clean across denoising. After each Euler
step the **denoised x0** is re-blended with the clean reference tokens before
the actual stepper update.

This is **inference only**. No training, no LoRA training paths, no dataset
code, no `peft`, no `bitsandbytes`. No `torch` in the runtime. Reference torch
code lives under `.references/DramaBox/` for reading; a separate dev venv
captures golden tensors at module boundaries for parity tests.

## Audit Corrections (rev1 → rev2 → rev3)

The plan has been revised twice based on Codex audits. The corrections that
landed in each revision:

### rev1 → rev2 (verified)

| Topic | rev1 (wrong) | rev2 (verified) |
| --- | --- | --- |
| DiT size | "22B audio-only" | **3.3B**, LTX-2.3-22B-derived |
| Latent shape | `[B, 8, T, 64]`, tokens `[B, T*F, C]` | `[B, 8, T, 16]`, tokens `[B, T, 128]` |
| Decoded mel bins | (conflated with latent) | **64**, output of `AudioVAE.decode` |
| Caption proj placement | inside DiT | **text path**, `caption_proj_before_connector=True` |
| Cross-attn KV cache | "cache once" | **cannot** — `cross_attention_adaln=True` |
| RoPE type | `interleaved` | `split` |
| Vocoder dtype | bf16 | **fp32 compute** (autocast) |

### rev2 → rev3 (this revision)

| Topic | rev2 (still wrong) | rev3 (verified) |
| --- | --- | --- |
| DiT output | "returns denoised x0" | **returns velocity**; `X0Model` wraps to compute `x0 = noisy - v * timesteps` |
| Tokenizer | "Gemma chat template + turn delimiters" | **plain stripped text**, `max_length=1024`, left padding, **no chat template** |
| Prompt pipeline | "three separate stages" | **one operation**: `EmbeddingsProcessor.process_hidden_states()` — feature extraction + additive-mask conversion + connector + binary-mask zeroing |
| FeatureExtractorV2 | "layer-selection recipe" | **stacks ALL 49 hidden states**, per-token RMSNorm, rescale `* sqrt(out_dim/embed_dim)`, then `audio_aggregate_embed` |
| audio_aggregate_embed | "small projection" | `Linear(188160 → 2048)` — **770 MB at bf16**; will be quantized to 4-bit MLX (~96 MB) |
| audio_embeddings_connector | "small connector module" | a 9-block 1D transformer with 128 learnable register tokens + per-head gated attention + 2-layer FFN (hidden=8192), output `[B, 128, 2048]` |
| Target frame math | "round(duration_s * fps)" | `n_frames = round(duration * fps) + 1`, aligned to `((n - 1 + 4) // 8) * 8 + 1`, then `AudioLatentShape.from_video_pixel_shape(...)` |
| Voice-ref token count | "10 s ≈ 50 tokens" | **10 s ≈ 250 tokens** (at fps=25 latent rate) |
| `post_process_latent` placement | "applied to state AFTER Euler step" | **applied to denoised x0 BEFORE Euler step** (`samplers._step_state:30-31`) |
| Rescale formula | `(cond.std/pred.std)**rescale` | **linear interp**: `factor = rescale * (cond.std/pred.std) + (1 - rescale); pred *= factor` |
| Audio-only RoPE | "3 axes with collapsed h/w" | **one temporal axis** `[B, 1, T, 2]` (start/end timings) |
| fp64 RoPE precision | "compute all RoPE in fp64" | **NumPy float64 grid only**, cast to float32 before sin/cos, output in compute dtype |
| audio_ff | "gated MLP" | **plain GELUApprox + Linear**; `apply_gated_attention=True` is per-head attention gates only (NOT a gated FFN) |
| AudioProcessor (mel front-end) | "part of VAE encoder" | **separate front-end module** for waveform→mel; lives upstream of `AudioVAE.encode` |
| Noiser formula | "additive" | **convex mix**: `noised = noise * scaled_mask + latent * (1 - scaled_mask)` where `scaled_mask = denoise_mask * noise_scale` |
| Attention mask convention | "raw [0,1]" | **[0,1] in state**, **converted to additive log-space** (`(mask - 1) * finfo.max`) before SDPA |
| Vocoder final activation | "tanh + clip" | **config-driven**: main has `final_activation`, BWE disables it, wrapper clamps residual + skip |

### rev3 → rev4 (this revision)

| Topic | rev3 (still wrong) | rev4 (verified) |
| --- | --- | --- |
| Connector block count | "9 blocks" | **8 blocks** (`transformer_1d_blocks.0..7`); reference default is 2, this checkpoint overrides to 8 |
| Connector block structure | "AdaLN + cross to registers + 2 FFN" | **`_BasicTransformerBlock1D`**: `rms_norm → self-attn → residual → rms_norm → ffn → residual`. No AdaLN, no scale-shift table, no cross-attention to registers. Just self-attention. |
| `learnable_registers` mechanism | "128 register output tokens" | **Replace padded slots** in the 1024-token text sequence. Tiled `1024/128 = 8×` so they fit. Non-padded slots keep audio_feats; padded slots take register values. Output mask becomes all-zero (all positions valid). |
| `a_ctx` shape | `[B, 128, 2048]` | **`[B, 1024, 2048]`** — same length as the padded Gemma input |
| `audio_aggregate_embed` plan | "must quantize to 4-bit MLX (~96 MB)" | **bf16 default**, matching reference. 4-bit is an experimental optimization gated by parity, not a v5 requirement. The earlier 4-bit math was wrong too (~193 MB at 4-bit affine on a 188160-input linear, not 96 MB). |
| Scheduler `tokens` | "T × F" (~5000 for 5 s) | **`math.prod(state.latent.shape[2:]) == 128`** — the patchified latent is `[B, T, 128]`, so shape[2:] is just the token dim. Sigma schedule is essentially **fixed per checkpoint**, not duration-scaled. |
| AdaLN modulation source | "timestep + caption_pool" | **Timestep only** for latent AdaLN. Cross-attention has a **separate prompt-sigma AdaLN** for K/V context modulation when `cross_attention_adaln=True`. No caption pool. |
| Connector memory | "0.6 GB at bf16" | **0.807 GB at bf16** (807 MB connector params per shard inspection) |
| AudioProcessor mel constants | "config-driven" | **Explicit**: `win_length=n_fft`, `hop_length=mel_hop_length`, `f_min=0`, `f_max=sr/2`, `window_fn=hann`, `center=True`, `pad_mode="reflect"`, `power=1.0`, `mel_scale="slaney"`, `norm="slaney"`, `log(clamp(spec, min=1e-5))`, output `[B, C, T, n_mels]` |

The rev1 corrections that **remain accurate** in rev4:

- Audio components prefix layout, including `audio_vae.per_channel_statistics`
  vs ignored `vae.per_channel_statistics` legacy alias
- `silence_latent_frame.pt` is training metadata, not runtime
- End-of-clip fix is length-gated (T > 513), not checkpoint-gated
- Defaults locked to warm-server (cfg=2.5, stg=1.5, rescale="auto", modality=1.0)
- Voice-ref parity captured at `--denoise_ref=False` (RE-USE out of scope)

The rev2 → rev3 corrections that audit pass 3 **verified directly**:

- DiT returns velocity (X0Model wraps), tokenizer is plain stripped text with
  no chat template, `process_hidden_states()` is the parity unit, all 49 hidden
  states stacked, feature rescale `* sqrt(2048/3840)`, target frame rounding,
  noiser convex mix, sampler `post_process_latent` BEFORE Euler step, guidance
  rescale linear interp, one-axis audio RoPE, fp64 RoPE-grid boundary, plain
  FFN, AudioProcessor separation, additive mask convention, config-driven
  vocoder activation

## Hard Constraints

Inherited from `CLAUDE.md`:

- pure MLX runtime — no torch, no torchaudio, no `mlx_lm`, no `transformers`
- end-to-end waveform output — no token-only path is acceptable
- `.safetensors` weights — already downloaded under `models/dramabox/`
- weights never in git
- local-path-first loading
- no `huggingface_hub` calls at runtime — local paths only

Specific to v5:

- inference only — no train/LoRA-fit code under `src/`
- 30-step Euler sampler at the warm-server defaults — the "proper settings"
- voice reference conditioning is part of the v5 baseline (raw-reference;
  RE-USE is out of scope)
- vocoder + BWE compute is **fp32** regardless of weight storage dtype
- `audio_aggregate_embed` (the 188160 → 2048 projection) is loaded **at bf16**
  matching the reference. 4-bit MLX quantization of this layer is reserved as
  an optional optimization measured against parity fixtures, NOT a default.
- KV cache scope: only the **`a_ctx` from the connector** (after the full
  prompt pipeline) is cached across the 30 steps. Cross-attention K/V
  projections inside the DiT are recomputed each step because
  `cross_attention_adaln=True` modulates context before projection.
- AdaLN modulation factors `(shift, scale, gate)` are functions of
  **timestep only** for the latent path — per-layer per-step constants.
  When `cross_attention_adaln=True`, a **separate prompt-sigma AdaLN** also
  modulates the cross-attention context before K/V projection. Neither
  modulation is shared across guidance passes (each pass uses its own
  context).

## Source Truth

Primary upstream runtime truth (line ranges cite the actual functions we port):

- `.references/DramaBox/src/inference_server.py` — warm-server entry. Target
  shape construction at lines 325-335 (frame rounding). "Proper settings"
  defaults at line 307-309.
- `.references/DramaBox/src/inference.py` — per-call CLI entry. **Different
  defaults** — secondary reference.
- `.references/DramaBox/src/audio_conditioning.py` — IC-LoRA append +
  asymmetric attention mask construction.
- `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/`
  - `tokenizer.py:18-24, 45-52` — `LTXVGemmaTokenizer` (plain text strip,
    left padding, max_length=1024 from caller, no chat template)
  - `feature_extractor.py:62-83, 112-141` — `FeatureExtractorV2` with
    per-token RMSNorm + rescale; `audio_aggregate_embed` is an `nn.Linear`
  - `embeddings_processor.py:15-19, 30-89` — `EmbeddingsProcessor` with the
    `process_hidden_states()` entry point; `convert_to_additive_mask`
  - `embeddings_connector.py` — `Embeddings1DConnector` (the connector
    transformer)
- `.references/DramaBox/ltx2/ltx_pipelines/utils/blocks.py:440-488` — how
  Gemma is loaded with `max_length=1024` and integrated as a text encoder
- `.references/DramaBox/ltx2/ltx_core/model/transformer/`
  - `model.py:389-430` — `LTXModel.forward` returns velocity
  - `model.py:461-486` — `X0Model.forward` converts velocity → x0 via
    `to_denoised(latent, v, timesteps)`
  - `transformer.py:391-398` — cross-attention AdaLN context-modulation
  - `attention.py:175-181, 240-252` — per-head `to_gate_logits` for gated
    attention
  - `attention.py:217-239` — `SKIP_AUDIO_SELF_ATTN` perturbation path
    (value passthrough, no QK softmax, `to_out` retained)
  - `feed_forward.py:6-15` — plain GELUApprox + Linear (no gated FFN)
  - `rope.py:69-87, 192-204` — NumPy float64 grid build, float32 sin/cos,
    output cast
  - `transformer_args.py:97-123` — multiplicative [0,1] mask → log-space
    additive conversion
- `.references/DramaBox/ltx2/ltx_pipelines/utils/samplers.py:20-31` — the
  step body: `denoised = post_process_latent(denoised, mask, clean)`
  BEFORE `stepper.step(state.latent, denoised, sigmas, idx)`
- `.references/DramaBox/ltx2/ltx_pipelines/utils/helpers.py:252-254` — the
  re-blend formula
- `.references/DramaBox/ltx2/ltx_pipelines/utils/denoisers.py:90-130` —
  batched-passes pattern (cond/uncond/ptb concat along batch)
- `.references/DramaBox/ltx2/ltx_core/components/`
  - `schedulers.py:21-57` — `LTX2Scheduler.execute` math
  - `guiders.py:258-268` — guidance formula + **linear** rescale
  - `noisers.py:23-35` — convex mix
  - `diffusion_steps.py:14-22` — Euler step with `to_velocity`
  - `patchifiers.py:282-348` — patchify/unpatchify + position grid
- `.references/DramaBox/ltx2/ltx_core/utils.py` — `to_velocity`, `to_denoised`
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/`
  - `audio_vae.py:262-272`, `ops.py:8-55` — `AudioProcessor` mel front-end
  - `vocoder.py:551-575` — fp32 autocast wrap
  - `model_configurator.py:25-38, 67-75` — config-driven final activation;
    BWE-specific disabled final activation
- `.references/DramaBox/ltx2/ltx_core/{tools.py,types.py,utils.py}` — types

Reference-only (do not vendor into runtime):

- `.references/DramaBox/ltx2/ltx_core/loader/*` — torch-shaped loader
- `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/encoders/*` — torch
  + `bitsandbytes` Gemma wrapper
- `.references/DramaBox/ltx2/ltx_core/quantization/fp8_*` — irrelevant on
  Apple Silicon
- `.references/DramaBox/ltx2/ltx_core/model/video_vae/*` and
  `model/upsampler/*` (except what BWE imports)

Checkpoint truth (verified shapes):

- `models/dramabox/dramabox-dit-v1.safetensors` — 1457 keys all under
  `model.diffusion_model.*`. 3.3B audio-only DiT.
- `models/dramabox/dramabox-audio-components.safetensors` — 1462 keys:
  - `vocoder.vocoder.*` (667) — main BigVGAN
  - `vocoder.bwe_generator.*` (557) — BWE BigVGAN
  - `vocoder.mel_stft.*` (3) — STFT precomputed bases
  - `audio_vae.encoder.*` (44) and `audio_vae.decoder.*` (56)
  - `audio_vae.per_channel_statistics.*` (2) — `mean-of-means`, `std-of-means`,
    both shape `[128]` (NB: 128 = `channels × mel_bins` = 8×16)
  - `vae.per_channel_statistics.*` (2) — legacy alias; **ignored**
  - `model.diffusion_model.audio_embeddings_connector.*` (129 keys):
    - `learnable_registers` `[128, 2048]` — 128 register tokens of width 2048
    - **8** × `transformer_1d_blocks.{i}.*` (i ∈ 0..7) each with ~16 keys:
      - `attn1.{q_norm.weight[2048], k_norm.weight[2048],
        to_gate_logits.{weight[32,2048], bias[32]},
        to_q.{weight[2048,2048], bias[2048]},
        to_k.{weight[2048,2048], bias[2048]},
        to_v.{weight[2048,2048], bias[2048]},
        to_out.0.{weight[2048,2048], bias[2048]}}`
      - `ff.net.0.proj.{weight[8192,2048], bias[8192]}`, `ff.net.2.{weight[2048,8192], bias[2048]}`
    - **No AdaLN, no scale-shift table, no cross-attention to registers.**
      Each block is `rms_norm → self-attn → residual → rms_norm → ffn → residual`.
    - One final `rms_norm` after the block stack (functional, no learnable params)
    - Total: 1 (registers) + 8 × 16 = 129 keys ✓
  - `text_embedding_projection.audio_aggregate_embed.{weight, bias}` —
    `weight [2048, 188160]`, `bias [2048]`. **770 MB at bf16.**
- `models/gemma_3_12b_it_4bit/` — our pure-MLX 4-bit Gemma 3 12B IT
  backbone (1300 keys, affine 4-bit group_size=64, ~6.2 GB)

DiT `__metadata__.config` verified flags:

```
caption_proj_before_connector = True
cross_attention_adaln         = True
apply_gated_attention         = True
rope_type                     = "split"
frequencies_precision         = "float64"
use_middle_indices_grid       = True
attention_type                = "default"
timestep_scale_multiplier     = 1000
audio_num_attention_heads     = 32
audio_attention_head_dim      = 64
audio_in_channels             = (absent; reference defaults to 128)
audio_out_channels            = 128
audio_cross_attention_dim     = 2048
num_layers                    = 48
audio_positional_embedding_max_pos = [20]
```

## Final Runtime Shape

```text
text prompt                                        voice ref WAV (optional)
    │                                                       │
    ▼                                                       ▼
LTXVGemmaTokenizer                                    AudioProcessor
  (strip, left-pad, max_length=1024, no chat tmpl)      (resample to 16 kHz,
    │                                                    MelSpectrogram,
    ▼                                                    log clamp, permute)
Gemma 3 IT MLX 4-bit forward                              │
  (output_hidden_states=True → list of 49 tensors)        ▼
    │                                                AudioVAE.encode
    ▼                                                  (per-channel-stat
EmbeddingsProcessor.process_hidden_states():            normalize)
                                                          │
  FeatureExtractorV2:                                     ▼
    encoded = stack(hidden_states, dim=-1)           ref_latent [B, 8, T_ref, 16]
       → [B, T, D=3840, L=49]                            │
    normed = per-token RMS norm (dim=2,                  │
             attention_mask zeroes pads)                 │
    normed.reshape([B, T, 3840*49 = 188160])             │
    rescaled = normed * sqrt(2048 / 3840)                │
    audio_feats = audio_aggregate_embed(rescaled)        │
       → [B, T, 2048]                                    │
                                                          │
  convert_to_additive_mask(attention_mask, dtype)         │
     → [-finfo.max where pad, 0 elsewhere]               │
                                                          │
  audio_embeddings_connector(audio_feats, add_mask)       │
    replace padded slots with tiled learnable_registers   │
      (128 registers × 8 tiles = 1024 fills)              │
    new attention_mask = all-zero (all slots valid)       │
    8 × (rms_norm + gated self-attn + ffn) blocks         │
    final rms_norm                                        │
     → audio_encoding [B, 1024, 2048]                     │
                                                          │
a_ctx [B, 1024, 2048]  (also a_ctx_neg for CFG)           │
    │                                                     │
    │                          target initial state ◄─────┘
    │                              + IC-LoRA append (asymmetric mask,
    │                                positions +0.5 s, strength=1.0)
    │                              + GaussianNoiser convex mix
    │                                (noise * mask + latent * (1-mask))
    │                              + denoise_mask: 0 over ref, 1 over target
    │                                  │
    │                                  ▼
    │                       noised state, latent [B, 8, T_total, 16],
    │                                  patchified to [B, T_total, 128]
    │                                  positions [B, 1, T_total, 2]
    │                                  │
    │                                  ▼
    │                       LTX2Scheduler.execute(steps=30, latent=...)
    │                       → sigmas[31]
    │                                  │
    └────────────────► cached a_ctx, a_ctx_neg
                              │
                              ▼
                       ┌── for each (sigma_i, sigma_{i+1}) ──┐
                       │   ┌──────── batched forward ──────┐ │
                       │   │ contexts concat along batch:  │ │
                       │   │   cond + uncond + ptb(STG)    │ │
                       │   │ states repeated to match      │ │
                       │   └───────────────┬───────────────┘ │
                       │                   ▼                 │
                       │   X0Model(velocity_model):          │
                       │     vx = LTXModel(...)              │
                       │       (48-block DiT; each block:    │
                       │        - timestep_embed → latent     │
                       │          AdaLN factors                │
                       │        - prompt-sigma AdaLN modulates │
                       │          cross-attn context per step  │
                       │        - audio_attn1 self (split RoPE│
                       │          on 1 temporal axis, fp64   │
                       │          grid → fp32 sin/cos,       │
                       │          per-head gated, optional   │
                       │          STG skip on block 29)      │
                       │        - audio_attn2 cross to a_ctx │
                       │        - audio_ff plain GELU+Linear)│
                       │       → velocity [B, T, 128]        │
                       │     denoised = to_denoised(latent,  │
                       │                vx, timesteps)       │
                       │     timesteps = sigma * denoise_mask│
                       │                   │                 │
                       │                   ▼                 │
                       │   chunk into cond / uncond / ptb    │
                       │   GuidedDenoiser merges via         │
                       │   MultiModalGuider:                 │
                       │     pred = cond + (cfg-1)(cond-unc) │
                       │          + stg*(cond-ptb)           │
                       │          + (mod-1)(cond-unc_mod)    │
                       │     if rescale != 0:                │
                       │       f = cond.std/pred.std         │
                       │       f = rescale*f + (1-rescale)   │
                       │       pred = pred * f               │
                       │   → denoised x0 [B, T, 128]         │
                       │                   │                 │
                       │                   ▼                 │
                       │   post_process_latent(denoised,     │
                       │     denoise_mask, clean_latent)     │
                       │   re-blends frozen ref tokens to    │
                       │   clean source BEFORE step          │
                       │                   │                 │
                       │                   ▼                 │
                       │   EulerDiffusionStep.step:          │
                       │     v = to_velocity(noisy, σ, x0)   │
                       │     new = noisy + v*(σ_next - σ)    │
                       │                                     │
                       │   mx.eval(state.latent) — bound the │
                       │   graph at the step boundary        │
                       │                                     │
                       └─────────────────────────────────────┘
                                  │
                                  ▼
                       clear_conditioning (strip ref tokens)
                                  │
                                  ▼
                       unpatchify [B,T,128] → [B,8,T,16]
                                  │
                                  ▼
                       end-of-clip silence-prior fix
                       (length-gated, T > 513)
                                  │
                                  ▼
                       audio_vae.per_channel_statistics
                       un-normalize
                                  │
                                  ▼
                       AudioVAE.decode → mel [B, 2, T_mel, 64]
                                  │
                                  ▼     (fp32 autocast wrap starts here)
                       main BigVGAN → 16 kHz stereo waveform
                                  │
                                  ▼     (still fp32)
                       BWE BigVGAN → 48 kHz stereo waveform
                                  │
                                  ▼     (config-driven final activation;
                                  │      BWE disables generator final
                                  │      activation, wrapper clamps)
                       cast to fp32, clip to [-1, 1]
                                  │
                                  ▼
                       output.wav  (48 kHz, 2 channels)
```

## Public API

New family lives at:

- `src/mlx_speech/models/dramabox/` — internals
- `src/mlx_speech/models/gemma3_text/` — shared MLX Gemma 3 backbone
  (text-only forward, returns ALL hidden states; reusable by any future
  family that wants Gemma 3 as a text encoder)
- `src/mlx_speech/generation/dramabox.py` — public wrapper

Public wrapper (mirrors `StepAudioEditXModel` shape):

```python
from mlx_speech.generation.dramabox import DramaBoxModel

model = DramaBoxModel.from_dir("models/dramabox")
# or
model = DramaBoxModel.from_paths(
    dit="models/dramabox/dramabox-dit-v1.safetensors",
    audio_components="models/dramabox/dramabox-audio-components.safetensors",
    gemma="models/gemma_3_12b_it_4bit",
)

result = model.generate(
    prompt='A woman speaks clearly, "The weather today will be sunny."',
    voice_ref=None,           # or path to WAV
    cfg_scale=2.5,
    stg_scale=1.5,
    rescale_scale="auto",     # warm-server default; cfg-aware schedule
    modality_scale=1.0,
    steps=30,
    seed=42,
    duration_s=5.0,           # or None to auto-estimate from prompt
)
# result.waveform : mx.array [2, T_samples] float32 in [-1, 1]
# result.sample_rate : 48000
```

Defaults are locked to the warm-server reference, not the per-call CLI:

| Setting | Warm-server (our default) | Per-call CLI (NOT our default) |
| --- | --- | --- |
| cfg_scale | 2.5 | 7.0 |
| stg_scale | 1.5 | 1.0 |
| stg_block | 29 | 29 |
| rescale_scale | "auto" | "auto" |
| modality_scale | 1.0 | 3.0 |
| steps | 30 | 30 |
| fps | 25.0 | 25.0 |
| seed | 42 | 42 |

## Implementation Stages

### Stage 1 — Plan and repo registration

- create this active plan (this file)
- update `CLAUDE.md` Plan Status: mark `plans/v5-dramabox.md` as **Active**
- add `docs/references.md` entries pinning the DramaBox upstream commit
- dev parity venv: **reuse `.venv-torch/`** (already at the repo root with
  torch==2.11.0, torchaudio, transformers==5.0.0, safetensors, einops, scipy,
  tokenizers — sufficient for captures). Document setup in
  `dev/parity/README.md`. (`.venv*/` is already in `.gitignore`.)
- no production code yet

### Stage 2 — Gemma 3 text backbone + DramaBox prompt pipeline

This is the largest stage by code surface and the trickiest by parity.
`audio_aggregate_embed` is 770 MB at bf16 — held at bf16 by default
(matches reference). On a 32 GB Mac the full warm-resident set fits comfortably
(memory budget below). A 4-bit MLX quantization of this layer is a follow-up
experiment gated by parity, not a v5 requirement.

#### 2a — Gemma 3 backbone (shared)

Land under `src/mlx_speech/models/gemma3_text/`:

- `config.py` — parse our `config.json` (text_config: hidden_size=3840,
  num_hidden_layers=48, num_attention_heads=16, num_key_value_heads=8,
  head_dim=256, rope_theta=1_000_000 global / 10_000 sliding,
  sliding_window=1024, vocab_size=262208, intermediate_size=15360)
- `model.py` — pure MLX Gemma 3 forward:
  - `Gemma3RMSNorm` (fp32 accumulation, weight in compute dtype)
  - `Gemma3RotaryEmbedding` — alternating global / sliding RoPE per layer
  - `Gemma3MLP` (gate_proj × up_proj, gelu_pytorch_tanh, down_proj)
  - `Gemma3Attention` GQA (16 q / 8 kv heads, head_dim=256), per-layer
    sliding-window mask
  - `Gemma3DecoderLayer` matching HF reference
  - `Gemma3Model.forward(input_ids, attention_mask, output_hidden_states=True)
    → tuple[mx.array, ...]` of 49 hidden states (embedding output + 48 layer
    outputs; in that order, matching transformers' convention)
- `checkpoint.py` — flat-key loader for `gemma_3_12b_it_4bit`; affine 4-bit
  weights via `mx.nn.QuantizedLinear`
- `tokenizer.py` — `LTXVGemmaTokenizer` equivalent:
  - thin wrapper over `tokenizers.Tokenizer` loaded from our `tokenizer.json`
  - `tokenize(text, max_length=1024) → (input_ids, attention_mask)`
  - `text = text.strip()`, no chat template, left padding, pad_token =
    eos_token if pad_token is None
  - returns `mx.array` not list-of-tuples (we don't need the upstream
    `tokenize_with_weights` plumbing)

#### 2b — DramaBox prompt pipeline

Land under `src/mlx_speech/models/dramabox/prompt/`:

- `feature_extractor.py` — `FeatureExtractorV2` port:
  - input: tuple of 49 hidden states, attention_mask [B, T=1024]
  - `encoded = mx.stack(hidden_states, axis=-1)` → `[B, T, 3840, 49]`
  - `variance = mean(encoded**2, axis=2, keepdims=True)` → `[B, T, 1, 49]`
  - `normed = encoded * mx.rsqrt(variance + 1e-6)` (per-token RMS norm)
  - `normed = normed.reshape(B, T, 3840 * 49)` → `[B, T, 188160]`
  - mask broadcast and zero-out padded positions
  - rescale: `normed = normed * mx.sqrt(2048 / 3840)` (where 2048 =
    `audio_aggregate_embed.out_features`, 3840 = embedding_dim)
  - apply `audio_aggregate_embed` at bf16 → `[B, T, 2048]`
- `aggregate_loader.py` — load `text_embedding_projection.audio_aggregate_embed`
  from audio-components at bf16. (4-bit MLX quantization is a follow-up
  optimization gated by parity, NOT a v5 requirement.)
- `additive_mask.py` — `convert_to_additive_mask(attention_mask, dtype)`:
  `(mask.astype(int64) - 1).astype(dtype).reshape(B, 1, 1, T) * finfo.max`
- `connector.py` — `Embeddings1DConnector` (port of
  `.references/.../embeddings_connector.py:72-198`):
  - learnable_registers buffer `[128, 2048]`
  - **`_replace_padded_with_learnable_registers(hidden_states, mask)`**:
    - `non_zero_count` = count of non-padded slots; pack non-padded values
      to the front of the sequence (rest zero-padded)
    - `flipped_mask = flip(binary_mask, axis=seq)` — moves padded slots to
      the back
    - `out = flipped_mask * adjusted + (1 - flipped_mask) * tiled_registers`
      — non-padded slots stay; padded slots take register values (tiled
      `1024 / 128 = 8` times)
    - new attention_mask = all-zero (additive 0 → no masking; all slots
      valid)
  - **8 × `_BasicTransformerBlock1D`**, each (no AdaLN, no scale-shift):
    ```
    h = h + attn1(rms_norm(h), mask, pe=rope_freqs)
    h = h + ff(rms_norm(h))
    ```
    where:
    - `attn1` is self-attention with `q_norm` + `k_norm` (RMSNorm on q/k),
      per-head `to_gate_logits[32, 2048]` (gated attention, sigmoid scalar
      gate per head), `to_q/k/v/out.0` all [2048, 2048] with bias
    - `ff` is `Linear(2048, 8192) → GELU → Linear(8192, 2048)` (plain, not
      gated)
  - one final `rms_norm` after the block stack (no learnable params)
  - RoPE: `LTXRopeType.split`, `frequencies_precision=float64`; positions
    are `arange(1024)`; `positional_embedding_max_pos = [4096]` (from
    `connector_positional_embedding_max_pos` in DiT metadata)
  - input: `audio_feats [B, 1024, 2048]`, additive_attention_mask
    `[B, 1, 1, 1024]`
  - output: `(audio_encoding [B, 1024, 2048], mask all-zero)`
- `processor.py` — `EmbeddingsProcessor.process_hidden_states` port:
  - one entry: `(hidden_states, attention_mask, padding_side="left") → a_ctx`
  - orchestrates feature_extractor → additive_mask → connector
- `prompt_encoder.py` — top-level orchestrator:
  - tokenize → Gemma forward (all hidden states) → processor → a_ctx

Validate:

- unit: Gemma config parsing, RoPE pattern, sliding-window mask
- unit: tokenizer produces fixed token ids + attention_mask for known input
  (left-padded to 1024)
- unit: additive mask shape + dtype + values for a known input
- unit: feature extractor RMS norm matches reference at fp32
- checkpoint: load MLX 4-bit Gemma; load aggregate (bf16);
  load connector (bf16); count keys
- runtime: short prompt forward against dev-venv torch reference:
  - cosine ≥ 0.999 on Gemma all-49 hidden states (Gemma 4-bit drift
    documented)
  - cosine ≥ 0.999 on the FeatureExtractorV2 output
  - cosine ≥ 0.999 on the aggregate output (bf16, parity-tight)
  - cosine ≥ 0.999 on the final `a_ctx [B, 1024, 2048]`

### Stage 3 — Latent grid, scheduler, noise, target shape

Target frame math (warm-server reference at `inference_server.py:325-335`):

```
n_frames = round(duration_s * fps) + 1
n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
pixel_shape = VideoPixelShape(batch=B, frames=n_frames, height=64, width=64,
                              fps=fps=25.0)
target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
# which expands to:
#   latents_per_second = 16000 / 160 / 4 = 25.0
#   audio_frames = round(n_frames/25.0 * 25.0)
#   channels = 8, mel_bins = 16
```

Land under `src/mlx_speech/models/dramabox/`:

- `shape.py` — `target_shape_from_duration(duration_s, fps=25.0)`:
  - implements the warm-server frame rounding above
  - returns `(B=1, C=8, T, F=16)` shape tuple
- `patchifier.py` — `AudioPatchifier(patch_size=1)`:
  - `patchify`: `[B, C, T, F] → [B, T, C*F]` via `b c t f → b t (c f)`
  - `unpatchify`: `[B, T, C*F] → [B, C, T, F]`
  - `get_patch_grid_bounds(target_shape, device)` → `[B, 1, T, 2]`
    (start/end timings per latent frame; one temporal axis)
- `scheduler.py` — `LTX2Scheduler.execute(steps=30, latent)`:
  - port `schedulers.py:21-57` exactly
  - **tokens = `math.prod(state.latent.shape[2:])`** — and since the latent
    is patchified by the time scheduler is called (`tools.py:186-190` calls
    `self.patchify(...)` inside `create_initial_state`), shape is `[B, T, 128]`
    so `shape[2:] = (128,)` and **tokens = 128 for this checkpoint** regardless
    of duration. The sigma schedule is essentially fixed per checkpoint.
  - max_shift=2.05, base_shift=0.95, BASE_ANCHOR=1024, MAX_ANCHOR=4096
  - exp time shift formula + stretch to terminal=0.1
  - returns `mx.array[float32, steps+1]`
- `noiser.py` — `GaussianNoiser(seed)`:
  - `mx.random.key(seed)` and `mx.random.normal(shape, dtype=fp32)` for noise
  - **convex mix** at `noisers.py:23-35`:
    `scaled_mask = state.denoise_mask * noise_scale`
    `noised = noise * scaled_mask + state.latent * (1 - scaled_mask)`
  - returns new state with `state.latent = noised`
- `state.py` — `LatentState` dataclass: `latent`, `denoise_mask`,
  `positions`, `clean_latent`, `attention_mask` (None or `[B, N, N]`)
- `tools.py` — `AudioLatentTools`:
  - `create_initial_state(target_shape)` — zero latent, ones denoise_mask,
    positions from patchifier `[B, 1, T, 2]`, clean_latent=zeros,
    attention_mask=None
  - `patchify_state(state)` / `unpatchify_state(state, target_shape)`
  - `clear_conditioning(state, num_target)` — slice the trailing ref tokens
  - `post_process_latent(denoised, mask, clean)` (matches
    `helpers.py:252-254`): `denoised = mask * denoised + (1 - mask) * clean`

Validate:

- unit: `target_shape_from_duration(5.0)` produces the same n_frames as the
  reference (verify against captured reference outputs)
- unit: patchify/unpatchify round-trip is identity
- unit: scheduler matches reference float32-tight for `tokens=128`
  (our actual case) AND for sanity values `tokens ∈ {1024, 4096}`
  exercising the shift anchors
- unit: noiser convex mix matches reference for fixed seed and known mask
- unit: post_process_latent leaves frozen ref tokens unchanged
  (denoise_mask=0 over them) and passes target through unchanged
  (denoise_mask=1)

### Stage 4 — AudioProcessor + AudioVAE (encoder + decoder)

The mel front-end is a **separate module** preceding the VAE encoder, not
part of the encoder. The reference path:

```
waveform [B, C, T_samples]   ←   loaded WAV (stereo or mono → broadcast to 2ch)
    │
    ▼
AudioProcessor (`audio_vae/audio_vae.py:262-272`, `ops.py:8-55`):
  resample → 16 kHz
  MelSpectrogram (config-driven: n_mels, win_length, hop_length, fmin, fmax,
                  pad, power, normalized)
  log clamp (log(spec + eps))
  permute to [B, C, T_mel, F_mel]
    │
    ▼
AudioVAE.encoder → [B, 8, T_latent, 16]
    │
    ▼
audio_vae.per_channel_statistics normalize
```

The decode path is the reverse:

```
latent [B, 8, T_latent, 16]
    │
    ▼
audio_vae.per_channel_statistics un-normalize
    │
    ▼
AudioVAE.decoder → [B, 2, T_mel, 64]   ← 64-mel-bin spectrogram, not waveform
    │
    ▼
(vocoder follows in Stage 5)
```

The VAE module structure is **config-driven**, not hardcoded. Stage 4 starts
by reading the audio-components shard keys + `model_configurator.py` to
recover the actual layer composition.

Land under `src/mlx_speech/models/dramabox/audio_vae/`:

- `audio_processor.py` — MLX/NumPy mel front-end (port of
  `audio_vae/ops.py:8-55`). **Exact parameters** (from audio-components
  metadata):
  - resample to `target_sample_rate = 16000`
  - **causal STFT** (the audio-components config sets `stft.causal = True`
    and `preprocessing.audio.causal_padding = 3`) — left-pad the input
    rather than reflect-padding centered, OR use the polyphase scheme the
    reference applies. Verify the exact padding via dev-venv capture.
  - `n_fft = filter_length = 1024`
  - `win_length = 1024`
  - `hop_length = 160`
  - `n_mels = 64` (the encoder input mel resolution; the 16-bin latent comes
    from VAE downsampling by 4×, not from this front-end)
  - `f_min = 0`, `f_max = 8000`
  - `window_fn = hann_window` (NumPy `numpy.hanning`)
  - `power = 1.0` (magnitude)
  - `mel_scale = "slaney"`, `norm = "slaney"`
  - log + clamp: `log(clamp(spec, min=1e-5))`
  - permute to `[B, C=2, T_mel, n_mels=64]`
- `config.py` — VAE config (resolved from audio-components metadata):
  - `double_z = True`, `z_channels = 8`, `in_channels = 2`, `out_ch = 2`
  - `ch = 128` (base channels), `ch_mult = [1, 2, 4]`
  - `num_res_blocks = 2`
  - `attn_resolutions = []` and `mid_block_add_attention = False` —
    **the VAE has NO attention layers**; drop any attention module from
    the port
  - `norm_type = "pixel"` — PixelNorm (`x / sqrt(mean(x**2, dim=channels) + eps)`)
  - `causality_axis = "height"` — height is the time axis for audio mel,
    so the causal Conv2d is causal along the height/time axis (full along
    width/frequency)
  - `dropout = 0.0` (inference, irrelevant)
- `causal_conv_2d.py` — causal Conv2d along height (time), full along
  width (frequency)
- `pixel_norm.py` — `PixelNorm`: normalize each spatial position
  independently across the channel axis
- `resnet.py` — VAE resblock (PixelNorm + activation + causal Conv2d)
- `downsample.py` / `upsample.py` — strided causal Conv2d variants. The
  encoder/decoder have 2 down/up stages each (3 levels via `ch_mult`)
  with `num_res_blocks=2` per level
- `encoder.py` — encoder forward: `[B, 2, T_mel, 64] → [B, 8, T_latent, 16]`.
  Spatial (height+width) downsample by 4× → 64/4 = 16 latent mel-bins.
  Time downsample governed by `downsample_time = False` per config — verify
  in implementation
- `decoder.py` — decoder forward: `[B, 8, T_latent, 16] → [B, 2, T_mel, 64]`
- `per_channel_statistics.py` — load and apply mean/std buffers
  (`audio_vae.per_channel_statistics.{mean-of-means, std-of-means}` both
  shape `[128]` — this is `channels * mel_bins = 8 * 16 = 128`)
- `model.py` — `AudioVAE` wiring

Validate:

- checkpoint: 44 encoder + 56 decoder + 2 per-channel-stat keys load with
  exact shapes
- unit: AudioProcessor on a known WAV matches dev-venv reference within
  fp32 epsilon on the mel spectrogram (resampling is the main source of
  drift; document the tolerance)
- runtime: encode dev-venv-captured waveform → cosine ≥ 0.9999 on latent
- runtime: decode dev-venv-captured latent → cosine ≥ 0.9999 on mel

### Stage 5 — Vocoder stack (main BigVGAN + BWE), fp32 compute

Both vocoders are BigVGAN-v2-style generators that **must run in fp32**
(`vocoder.py:551-575`).

The final activation is **config-driven**:

- main vocoder: has a final activation per `model_configurator.py:25-38`
- BWE generator: disables its own final activation
  (`model_configurator.py:67-75`)
- wrapper clamps residual + skip outputs (`vocoder.py:590-594`)

Land under `src/mlx_speech/models/dramabox/vocoder/`. **Both vocoders have
explicit config** from the audio-components metadata:

**Main vocoder** (mel → 16 kHz wav):
- `upsample_initial_channel = 1536`
- `upsample_rates = [5, 2, 2, 2, 2, 2]` (product = 160 = mel hop)
- `upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]`
- `resblock = "AMP1"`, `resblock_kernel_sizes = [3, 7, 11]`,
  `resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]`
- `activation = "snakebeta"`
- `use_tanh_at_final = False`, `use_bias_at_final = False`
- input: mel `[B, 2, T, 64]`, output: `[B, 2, T*160]` at 16 kHz

**BWE** (16 kHz wav → 48 kHz wav, via its own mel re-extraction):
- has its **own STFT/mel front-end** inside: `n_fft = 512`, `win_size = 512`,
  `hop_length = 80`, `num_mels = 64`
- `upsample_initial_channel = 512`
- `upsample_rates = [6, 5, 2, 2, 2]` (product = 240 = 80 hop × 3× rate ratio)
- `upsample_kernel_sizes = [12, 11, 4, 4, 4]`
- `resblock = "AMP1"`, same kernel/dilation sizes as main
- `activation = "snakebeta"`
- `apply_final_activation = False`, `use_tanh_at_final = False`,
  `use_bias_at_final = False`
- `input_sampling_rate = 16000`, `output_sampling_rate = 48000`
- input: 16 kHz wav `[B, 2, T_16k]`, output: `[B, 2, T_48k]`

Module breakdown:

- `snakebeta.py` — `snake_beta(x, alpha, beta) = x + (1/(beta+eps)) *
  sin(alpha*x)**2`. Performance-critical.
- `amp_block.py` — `AMPBlock1` (3-resblock variant): dilated conv stack
  with snakebeta. Kernel sizes 3/7/11, dilations [1,3,5] per resblock.
- `bigvgan.py` — main generator: conv pre + 6 upsample stages (transposed
  conv + AMPBlock1×3) + conv post. No tanh, no final bias.
- `bwe.py` — BWE generator: own mel STFT front-end + 5 upsample stages.
  Inherits the same AMPBlock1 backbone. Final activation explicitly off.
- `mel_stft.py` — STFT mel front-end used by BWE; load `vocoder.mel_stft.*`
  buffers. STFT params: n_fft=512, hop=80, win=512.
- `pipeline.py` — `VocoderPipeline.forward(latent_mel) → 48 kHz wav`:
  - fp32 wrap (`mx.array.astype(mx.float32)` on input + per-op upcast of
    bf16 weights, or store weights as fp32 if memory permits)
  - main BigVGAN → 16 kHz
  - BWE BigVGAN (re-mel + upsample) → 48 kHz
  - clip residual + skip outputs per the wrapper

Validate:

- checkpoint: 667 main + 557 BWE + 3 mel_stft keys load
- runtime: dev-venv-captured mel → main BigVGAN; PSD within 0.5 dB across
  16 kHz output
- runtime: BWE 16 → 48 kHz on a real 16 kHz sample; PSD within 0.5 dB

### Stage 6 — LTX DiT (audio-only, 48 layers, 3.3B params), velocity model

The DiT is the **velocity model**: it returns `vx` (predicted velocity at
the current sigma). `X0Model` in Stage 7 wraps it to produce `x0` via
`to_denoised`.

Configuration must come from the DiT shard `__metadata__.config` (the
verified flags are listed above).

#### 6a — Primitives

- `rope.py` — `LTXRopeType.split` RoPE. For audio-only mode there is **one
  temporal axis** with positions `[B, 1, T, 2]` (start/end timings per
  frame). The "split" variant divides head_dim across the position-vector
  components, NOT interleaved. Frequencies built via **NumPy float64 grid**,
  indices cast to float32 before sin/cos, output in compute dtype.
- `attention.py` — supports:
  - self-attention (`audio_attn1`) with RoPE, optional asymmetric `[B,N,N]`
    mask (IC-LoRA case), optional `SKIP_AUDIO_SELF_ATTN` perturbation flag
    per-block, **per-head gated attention** via `to_gate_logits` (32-dim
    output, one logit per head, sigmoid → per-head scalar gate applied to
    attention output)
  - cross-attention (`audio_attn2`) with AdaLN-modulated context input
    (K/V projection runs each step), per-head gated attention
  - mask conversion: state stores `[0,1]` multiplicative, converted to
    log-space additive (`(mask - 1) * finfo.max`) before SDPA
  - `mx.fast.scaled_dot_product_attention` where the mask shape allows;
    explicit softmax fallback for the asymmetric IC-LoRA mask
- `adaln.py` — produces per-block factors. With `cross_attention_adaln=True`
  there are additional `(shift, scale)` for the cross-attention CONTEXT
  input (applied to context tensor before its K/V projection).
- `timestep_embedding.py` — sinusoidal embedding + 2-layer MLP, with
  `timestep_scale_multiplier=1000`
- `feed_forward.py` — **plain** `GELUApproxFF`: `Linear(2048, hidden) →
  GELU(approx="tanh") → Linear(hidden, 2048)`. **NOT a gated MLP.**
- `gated_attention.py` — the per-head gating helper applied inside
  `attention.py`. NOT a separate module on the block level.

#### 6b — Block + model

- `block.py` — one `LTXBlock`:
  - **timestep-only AdaLN** for the latent path: `timestep_embed(t)` →
    per-block (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    gate_mlp). No caption pooling.
  - **separate prompt-sigma AdaLN** for cross-attention context (when
    `cross_attention_adaln=True`): produces `(ca_shift, ca_scale)` applied
    to the context tensor before its K/V projection
  - audio_attn1 self with optional asymmetric mask + optional STG skip,
    per-head gated, latent-AdaLN-modulated input + residual gate
  - audio_attn2 cross to ca-modulated context, per-head gated,
    latent-AdaLN-modulated input + residual gate
  - audio_ff plain MLP (GELU+Linear), latent-AdaLN-modulated input +
    residual gate
- `model.py` — `LTXModel.audio_only`:
  - input proj: `audio_in_channels=128 → hidden=2048`
  - 48 × `LTXBlock`
  - final AdaLN + output proj: `2048 → audio_out_channels=128`
  - **returns velocity** in patch-token shape `[B, T, 128]`

#### 6c — Checkpoint loading

- `checkpoint.py` — load `dramabox-dit-v1.safetensors`:
  - strip `model.diffusion_model.` prefix → flat keys
  - per-layer mapping reconstructed from the actual reference layout in
    `.references/DramaBox/ltx2/ltx_core/model/transformer/transformer.py`
    BEFORE writing the loader
  - assert: missing/unexpected key sets are empty
  - IC-LoRA pre-merged in the dev checkpoint per `config.json` — assert no
    LoRA-named keys

Validate:

- unit: RoPE positions for known audio target shape match reference at fp32
- unit: gated attention applies per-head scalar gate correctly
- unit: AdaLN factors include the cross-attn context modulation triple when
  `cross_attention_adaln=True`
- unit: STG skip leaves `to_out` + residual + AdaLN gate intact
- checkpoint: all 1457 DiT keys map cleanly
- runtime: single-step forward `(noisy, sigma, a_ctx) → vx` on captured
  input; cosine ≥ 0.999 on velocity output

### Stage 7 — Guidance, denoising loop, IC-LoRA conditioning

Land under `src/mlx_speech/models/dramabox/diffusion/`:

#### X0Model wrapper + Euler step

- `x0_model.py` — `X0Model(velocity_model)`:
  - calls `vx = velocity_model(...)`
  - returns `denoised = to_denoised(latent, vx, timesteps)`
  - `timesteps = sigma * denoise_mask` (per `model.py:472-485`)
  - `to_denoised(x, v, t) = x - v * t` (verify exact formula against
    `ltx_core/utils.py`)
- `diffusion_step.py` — `EulerDiffusionStep.step`:
  - `velocity = to_velocity(noisy, sigma, denoised) = (noisy - denoised) / sigma`
  - `new_sample = noisy + velocity * (sigma_next - sigma)` in fp32, cast
    back to compute dtype

#### Denoiser

- `denoiser.py` — `GuidedDenoiser(a_context, audio_guider, negative_ctx)`:
  - **returns denoised x0**, not velocity
  - builds the batched-passes list (cond / uncond / ptb), concatenates
    contexts along batch dim, repeats states, calls X0Model **once** per
    step, chunks results, merges via `MultiModalGuider`

#### Guidance and STG

- `guider.py` — `MultiModalGuider(cfg_scale, stg_scale, stg_blocks,
  rescale_scale, modality_scale, negative_context)`:
  - guidance vector:
    `pred = cond + (cfg-1)*(cond-uncond) + stg*(cond-uncond_ptb) +
           (modality-1)*(cond-uncond_modality)`
  - rescale (linear interpolation, **not exponentiation**):
    ```
    if rescale_scale != 0:
        factor = cond.std() / pred.std()
        factor = rescale_scale * factor + (1 - rescale_scale)
        pred = pred * factor
    ```
- `auto_rescale.py` — port `auto_rescale_for_cfg` verbatim from
  `inference_server.py:91-116`
- `perturbations.py` — `PerturbationType.SKIP_AUDIO_SELF_ATTN` threaded
  through DiT attention. At active blocks, self-attention runs value
  projection only (no Q×K softmax), but the result still goes through
  `to_out`, residual, AdaLN gate.

#### IC-LoRA conditioning

- `conditioning.py` — `AudioConditionByReferenceLatent.apply_to`:
  - patchify the encoded ref latent
  - compute positions via patchifier's `get_patch_grid_bounds`, add +0.5 s
    offset
  - denoise_mask = `1 - strength` (strength=1.0 → mask=0 → frozen)
  - asymmetric attention mask `[B, N+M, N+M]`:
    - target → target: inherits prior mask (None → all-ones)
    - target → ref: 1
    - ref → target: 0
    - ref → ref: 1
  - extend state.latent / denoise_mask / positions / clean_latent /
    attention_mask along the token axis
  - tracks `(num_target, num_ref)` for `clear_conditioning`

#### Loop

- `sampler.py` — `euler_denoising_loop(sigmas, state, stepper, transformer,
  denoiser)` matching `samplers.py:20-31`:
  ```
  for i in range(len(sigmas) - 1):
      denoised = denoiser(state.latent, sigmas[i])          # X0Model output
      denoised = post_process_latent(denoised,              # re-blend ref
                                     state.denoise_mask,
                                     state.clean_latent)
      state.latent = stepper.step(state.latent, denoised,   # Euler step
                                  sigmas, i)
      mx.eval(state.latent)
  return state
  ```

- `silence_prior_fix.py` — frames-512-513 linear interpolation when
  `T > 513`. Length-gated, not checkpoint-gated.

Memory discipline:

- per-step eval is the right outer boundary (batched cond/uncond/ptb is one
  MLX graph; chunking happens after the forward)
- inside the DiT 48-block forward, MLX lazy eval keeps the working set
  bounded by 1-2 blocks at a time; add `mx.eval` inside the block loop only
  under memory pressure
- explicit `mx.eval` at: (a) after prompt encoding (cache `a_ctx`,
  `a_ctx_neg`), (b) after each Euler step (`mx.eval(state.latent)`),
  (c) after VAE decode, (d) after vocoder, (e) after BWE

Validate:

- unit: `to_velocity` and `to_denoised` formulas match reference
- unit: rescale linear interp matches reference for several
  `(cond.std, pred.std, rescale_scale)` triples
- unit: `auto_rescale_for_cfg` matches reference for cfg ∈
  {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0, 10.0}
- unit: silence-prior fix is a no-op for `T ≤ 513`, mutates exactly two
  frames for `T > 513`
- unit: asymmetric mask for `(num_target, num_ref) = (313, 250)` matches
  reference exactly
- runtime: 30-step Euler loop on `(prompt, seed, no_ref)` produces
  cosine ≥ 0.999 on the final pre-VAE latent vs reference
- runtime: 30-step loop with voice ref produces cosine ≥ 0.999 on
  the final post-clear_conditioning latent

### Stage 8 — Wrapper, CLI, conversion

Land:

- `src/mlx_speech/generation/dramabox.py` — `DramaBoxModel`:
  - `from_dir(root)` discovers DiT + audio-components + Gemma 4-bit
  - `from_paths(...)`
  - `.generate(prompt, voice_ref=None, **settings) → DramaBoxResult`
  - returns `DramaBoxResult(waveform, sample_rate, duration_s,
    settings_used)`
  - file I/O lives in the CLI, not the wrapper
  - defaults are the **warm-server** values
- `scripts/generate_dramabox.py` — CLI matching the wrapper
- `scripts/convert_dramabox.py` — optional MLX-native re-pack (lower
  priority; only after parity holds)

Smoke test (gated on checkpoint presence):

- `tests/runtime/test_dramabox_smoke.py`: 5 s prompt, no voice ref, 30
  steps, asserts non-NaN waveform with shape `[2, ~240_000]`
- gated `tests/integration/test_dramabox_long.py` with `RUN_LOCAL_INTEGRATION=1`:
  real prompt + voice ref, writes WAV, checks PSD vs recorded reference

### Stage 9 — Docs + release readiness

- `docs/dramabox.md` — model family guide (architecture, settings, weights,
  voice ref usage, known limitations including aggregate-quantization drift)
- update `README.md` model-family list
- `docs/references.md` — pin DramaBox upstream commit + LTX-2 commit
- the v5 plan flips from `Active` to `Done`

## Dependency Stance

Runtime (`src/mlx_speech/`):

- `mlx`, `numpy`, `safetensors`, `soundfile`, `tokenizers` — already in
  `pyproject.toml`; sufficient
- nothing new required for v5

Dev / parity (`.venv-dev/`, not packaged):

- `torch==2.8.0`, `transformers`, `safetensors`, vendored `ltx_core`,
  `einops`, `scipy`
- fixture capture scripts under `dev/parity/dramabox/`:
  - `capture_gemma.py` — text → all 49 hidden states
  - `capture_prompt.py` — text → `a_ctx` (FeatureExtractorV2 + aggregate +
    connector); captures intermediate at each boundary
  - `capture_vae.py` — WAV ↔ latent (encoder + decoder)
  - `capture_audio_processor.py` — WAV → mel front-end output
  - `capture_bigvgan.py` + `capture_bwe.py`
  - `capture_dit_velocity.py` — `(noisy, sigma, a_ctx) → velocity` for one
    block and for the full DiT
  - `capture_full_pipeline.py` — end-to-end with fixed seed,
    `--denoise_ref=False` for the voice-ref case
- fixtures at `tests/fixtures/dramabox/*.safetensors`, max ~100 MB total
- dev venv is **never** imported by anything under `src/mlx_speech/`

## Memory Budget (32 GB Mac target)

Live model footprint at runtime:

| Component | Storage | Compute |
| --- | ---: | ---: |
| Gemma 3 12B IT (MLX 4-bit) | 6.2 GB | 6.2 GB |
| `audio_aggregate_embed` (bf16) | 0.77 GB | 0.77 GB |
| `audio_embeddings_connector` (bf16) | 0.81 GB | 0.81 GB |
| DiT (bf16) | 6.58 GB | 6.58 GB |
| AudioProcessor + AudioVAE (bf16) | 0.5 GB | 0.5 GB |
| Main BigVGAN + BWE (bf16 storage, fp32 compute) | 0.8 GB | 1.6 GB (transient) |
| Per-step buffers (3 guidance passes × ~150 MB) | — | 0.45 GB (transient) |
| **Total persistent** | **~15.7 GB** | — |
| **Peak during generate** | — | **~17.5 GB** |

This is comfortable on 32 GB. Tight on 24 GB (would need to evict modules
between phases, or hold only the active subset warm). Out of scope for 24 GB:
v5 ships for 32 GB+ Macs.

## Validation

Test tiers stay aligned with `CLAUDE.md`:

- `tests/unit/` — config parsing, RoPE math, sigma math, patchifier round-trip,
  asymmetric mask construction, auto-rescale schedule, silence-prior fix,
  post_process_latent re-blend, to_velocity/to_denoised math, additive mask
  conversion, noiser convex mix, wrapper-arg handling
- `tests/checkpoint/` — Gemma backbone load, DiT load, audio VAE load, main
  BigVGAN load, BWE load, connector load, aggregate load (bf16)
- `tests/runtime/` — module-by-module parity against captured fixtures
- `tests/integration/` — `RUN_LOCAL_INTEGRATION=1` only:
  - one no-ref smoke (~5 s output)
  - one with-ref smoke (~5 s output, 10 s reference, no RE-USE)

Before reporting a stage complete:

- `pytest tests/unit/` passes
- the checkpoint and runtime tests for that stage's subsystem pass
- `RUN_LOCAL_INTEGRATION=1 pytest tests/integration/` is run manually at
  the end of Stage 8 and again at the end of Stage 9

## Assumptions and defaults

- "Proper settings" = warm-server defaults: `cfg=2.5 stg=1.5 stg_block=29
  rescale=auto modality=1.0 steps=30 fps=25 seed=42 dtype=bf16-storage /
  fp32-vocoder`
- v5 ships **only the dev checkpoint path**
- voice reference duration is fixed at 10 s for v5; parity measured at
  `--denoise_ref=False` (RE-USE out of scope)
- no `torch.compile` equivalent in v5 — MLX lazy eval + explicit `mx.eval`
  boundaries is the optimization story
- Gemma backbone is pure-MLX 4-bit affine (group_size=64). Drift vs bf16
  documented in fixtures
- `audio_aggregate_embed` is loaded at **bf16** matching reference (770 MB).
  4-bit MLX quantization of this layer is a follow-up optimization gated by
  parity fixtures, not a v5 default
- 32 GB Mac target. 24 GB out of scope for v5

## Out of Scope

- training, LoRA fit, dataset construction
- distilled-checkpoint defaults (not released by Resemble)
- streaming generation (chunked or token-by-token)
- video-conditioned audio generation (LTX-2 multi-modal path)
- `nvidia/RE-USE` reference denoiser
- Perth watermark
- `silence_latent_frame.pt` (training metadata only)
- `vae.per_channel_statistics` legacy alias keys
- 24 GB Mac support
