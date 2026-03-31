# v2: VibeVoice Large — Hybrid LLM+Diffusion TTS

**Status: DONE** (2026-03-30)

## Scope

Port Microsoft's VibeVoice Large (9B) to pure MLX. This is a fundamentally
different architecture from MOSS-TTS: continuous VAE latents + per-frame
diffusion instead of discrete RVQ codes.

VibeVoice is the second model family in mlx-speech, validating the library's
model adapter design.

## What Was Delivered

- Full VibeVoice Large 9B in pure MLX (~2700 lines)
- Qwen2 backbone (28 layers, GQA 28h/4kv), diffusion head (4 layers,
  DPM-Solver++), causal conv VAE codec (7 stages, 3200x compression),
  semantic feedback encoder, speech connectors
- Int8 quantized weights (9.4 GB from 18.7 GB BF16)
- Voice cloning from reference audio
- Multi-speaker dialogue up to 4 speakers in single generation pass
- Streaming conv cache for acoustic decoder and semantic encoder
- Adaptive diffusion step scheduling (opt-in, 20→10 after warmup)
- 38 tests passing, ruff clean
- Scripts: generate_vibevoice.py, convert_vibevoice.py

## Performance

- ~1.5x real-time without voice cloning (int8, 20 diffusion steps)
- ~0.6-0.8x real-time with voice cloning (reference prompt overhead)
- Natural stop behavior (model emits EOS)

## Known Limitations

- No sound effects / Foley / music (model limitation per Microsoft)
- Singing and emotion are emergent, not explicitly controllable
- Voice cloning adds latency (reference frames in attention context)
- Concat KV cache (pre-allocated was slower in MLX's lazy eval model)
- Batch audio decode at generation end (no per-frame streaming output)

## Pipeline

```
text with speaker tags (Speaker 1:, Speaker 2:, ...)
  + per-speaker reference audio
  → acoustic tokenizer encode (waveform → 64-dim VAE latent)
  → processor (format text, build voice prompt, splice embeddings)
  → Qwen2 autoregressive loop:
      for each token:
        if control token == speech_diffusion:
          → diffusion head (20 DPM-Solver steps → 64-dim latent)
          → acoustic tokenizer decode (latent → audio chunk, streaming)
          → semantic tokenizer encode (audio chunk → 128-dim feedback)
          → inject acoustic + semantic embeddings for next step
  → concatenate audio chunks → 24 kHz waveform output
```

## Architecture

### Key Difference from MOSS-TTS

| | MOSS-TTS | VibeVoice |
|---|---|---|
| Audio representation | Discrete RVQ codes (16-32 codebooks) | Continuous 64-dim VAE latents |
| LM output | Predicts audio tokens via heads | Predicts 4 control tokens only |
| Audio generation | LM heads → codec decode | Per-frame diffusion → conv VAE decode |
| Codec | Cat codec (LFQ, ~75 Hz) | Causal conv VAE (7.5 Hz) |
| Feedback | None | Semantic re-encoding → next LM step |
| Backbone | Qwen3 | Qwen2 (Qwen2.5-7B) |
| Params | 1.7B / 8B | 9.34B |

### Components

```
VibeVoiceForConditionalGeneration
├── language_model: Qwen2Model        # 28 layers, hidden 3584, GQA 28h/4kv
│   └── lm_head: Linear(3584, 152064) # text + 4 control tokens
├── acoustic_tokenizer                 # causal conv VAE
│   ├── encoder: 7 stages, ratios [8,5,5,4,2,2] = 3200x downsample
│   │   └── Block1D (depthwise conv + FFN + layer-scale)
│   └── decoder: mirror structure, ConvTranspose1d upsampling
│       └── streaming via per-layer causal conv cache
├── semantic_tokenizer                 # encoder only, deterministic
│   └── encoder: same architecture, vae_dim=128, no stochasticity
├── acoustic_connector: Linear(64, 3584) → RMSNorm → Linear(3584, 3584)
├── semantic_connector: Linear(128, 3584) → RMSNorm → Linear(3584, 3584)
└── prediction_head (diffusion)        # 4-layer adaLN FFN
    ├── HeadLayer × 4: adaLN modulation + SwiGLU FFN
    ├── FinalLayer: → latent_size (64)
    └── DPMSolverMultistepScheduler: 20 steps, v-prediction, cosine β
```

### Control Tokens

The LM only ever produces these four tokens:
- `<|vision_start|>` — speech segment start
- `<|vision_end|>` — speech segment end
- `<|vision_pad|>` — triggers one diffusion decode frame
- `<|endoftext|>` — EOS

No audio tokens exist in the vocabulary. All audio is generated out-of-band
by the diffusion head.

### Voice Cloning

No speaker ID embedding or lookup table. Speaker identity is communicated
entirely through continuous acoustic latent embeddings of reference audio,
spliced into the input embedding sequence at prefill time.

### Diffusion Decode (per frame)

Each time the LM emits `<|vision_pad|>`:
1. Take the LM's last hidden state as condition
2. Start from random noise (64-dim)
3. Run 20 DPM-Solver denoising steps (v-prediction, cosine schedule)
4. Classifier-Free Guidance (cfg_scale=3.0, negative from speech_start)
5. Result: one 64-dim acoustic latent
6. Decode via acoustic tokenizer decoder (streaming, causal conv cache)
7. Re-encode via semantic tokenizer → 128-dim
8. Project both through connectors, sum, inject as next step's embedding

## Configuration (from checkpoint)

| Parameter | Value |
|-----------|-------|
| `model_type` | `vibevoice` |
| LM hidden_size | 3584 |
| LM num_layers | 28 |
| LM num_attention_heads | 28 |
| LM num_key_value_heads | 4 (GQA) |
| LM intermediate_size | 18944 |
| LM vocab_size | 152064 |
| LM max_position_embeddings | 32768 |
| LM rope_theta | 1000000 |
| acoustic vae_dim | 64 |
| semantic vae_dim | 128 |
| encoder_ratios | [8, 5, 5, 4, 2, 2] |
| encoder_depths | 3-3-3-3-3-3-8 |
| diffusion head_layers | 4 |
| diffusion head_ffn_ratio | 3.0 |
| ddpm_num_inference_steps | 20 |
| ddpm_beta_schedule | cosine |
| prediction_type | v_prediction |
| sampling_rate | 24000 |
| target_dB_FS | -25 |
| speech_tok_compress_ratio | 3200 |
| total params | ~9.34B (BF16) |
| weight shards | 10 × safetensors, ~18.7 GB |
| text tokenizer | Qwen2.5-7B BPE (vocab 152064) |

## Prerequisites

The v1 MOSS-TTS runtime provides reusable patterns:
- quantized weight loading from local safetensors
- `W8Abf16` mixed-precision runtime
- KV cache design patterns
- processor conversation building
- waveform I/O and normalization

What's new and cannot be reused from v1:
- causal convolutional VAE (encoder + decoder + streaming cache)
- DPM-Solver diffusion scheduler
- Qwen2 backbone (adapt from existing Qwen3 — minor differences)
- per-frame diffusion decode loop interleaved with autoregressive LM
- semantic feedback injection

## Implementation Stages

### Stage 1 — Checkpoint + Config

- download VibeVoice-Large weights from HuggingFace
- parse `config.json` into MLX-facing `VibeVoiceConfig` with sub-configs
- inspect weight index (`model.safetensors.index.json`) for key layout
- map checkpoint keys to planned MLX module tree
- write `sanitize()` for key remapping
- **Test:** load weights, verify no missing/unexpected keys

### Stage 2 — Qwen2 Backbone

- adapt existing Qwen3 implementation for Qwen2 differences:
  - no sliding window attention (this config)
  - different RoPE config (theta=1e6, no scaling)
  - different norm eps (1e-6 vs Qwen3's value)
  - GQA with 28 heads / 4 KV heads
- `lm_head` projects to vocab_size (152064)
- **Test:** load real weights → random input → verify logit shapes

### Stage 3 — Acoustic Tokenizer (Conv VAE)

This is the largest new component.

- `Block1D`: depthwise conv → pointwise (FFN) → layer-scale → residual
- `TokenizerEncoder`: stem Conv1d → 7 downsampling stages
  - each stage: strided Conv1d + Block1D × depth
  - ratios [8,5,5,4,2,2], depths [3,3,3,3,3,3,8]
  - output: 64-dim mean (fixed σ=0.5 VAE)
- `TokenizerDecoder`: mirror with ConvTranspose1d upsampling
  - streaming support via per-layer causal conv cache
- all Conv1d must be causal (left-padded) for streaming
- `SConv1d` / `SConvTranspose1d` wrappers for causal padding + cache
- **Test:** encode → decode round-trip on random audio, verify shapes

### Stage 4 — Semantic Tokenizer

- same encoder architecture as acoustic, but:
  - vae_dim=128, no stochasticity (fix_std=0, no decoder)
  - deterministic: just the mean, no sampling
- **Test:** encode random audio → verify output shape (T', 128)

### Stage 5 — Connectors + Diffusion Head

- `SpeechConnector`: Linear → RMSNorm → Linear (two instances)
- `VibeVoiceDiffusionHead`:
  - `HeadLayer` × 4: adaLN modulation (SiLU → Linear → shift/scale/gate)
    + SwiGLU FFN (gate_proj, up_proj, down_proj)
  - `FinalLayer`: adaLN → Linear → 64-dim output
  - `cond_proj`: Linear(LM_hidden → head_hidden) for conditioning
- `DPMSolverMultistepScheduler`: port from reference impl
  - cosine β schedule, v-prediction, 20 inference steps
  - pure math — no framework-specific ops
- **Test:** load weights → verify head output shape, scheduler step shapes

### Stage 6 — Generation Loop

The most complex stage. Interleaves autoregressive LM with per-frame diffusion.

- prefill: encode reference audio → splice acoustic embeddings into input
- per-step:
  1. LM forward → logits over 4 control tokens
  2. sample token
  3. if `speech_diffusion`: run diffusion head (20 denoising steps)
  4. decode acoustic latent → audio chunk (streaming conv decoder)
  5. re-encode via semantic tokenizer → 128-dim
  6. project acoustic + semantic through connectors
  7. sum → next step's input embedding
- CFG: negative condition from speech_start position
- stop: `endoftext` token
- **Test:** generate from loaded model → verify token sequence + audio chunks

### Stage 7 — Int8 Conversion

- 9.34B params benefits substantially from int8 (~9.3 GB vs ~18.7 GB)
- quantize Qwen2 backbone (linear layers in attention + MLP)
- acoustic/semantic tokenizer conv layers: evaluate whether int8 helps
  (conv layers may need float for quality)
- diffusion head: likely keep float (small, 4 layers)
- **Test:** convert → load → verify shapes, run smoke generation

### Stage 8 — End-to-End + CLI

- wire: text → processor → generate → WAV
- voice cloning: reference audio → acoustic encode → embed splice
- multi-speaker: up to 4 speakers with "Speaker N:" format
- CLI:
  ```
  mlx-speech vibevoice \
    --text "Speaker 1: Hello! Speaker 2: Hi there!" \
    --speaker1-audio s1.wav \
    --speaker2-audio s2.wav \
    -o output.wav
  ```
- **Test:** produce multi-speaker WAV from real checkpoint

## Checkpoint Layout

```
models/vibevoice/
  original/     ← VibeVoice-Large safetensors (10 shards, ~18.7 GB)
  mlx-int8/     ← converted MLX int8 weights
```

## Done Criteria

v2 is done when:
- VibeVoice Large generates speech from text input on pure MLX
- voice cloning works via reference audio
- multi-speaker dialogue (up to 4 speakers) produces distinct voices
- streaming acoustic decode via causal conv cache
- runs entirely on MLX with no torch dependency
- loads from converted int8 MLX weights
- output is 24 kHz waveform
- CLI provides single-speaker and multi-speaker modes

## Out of Scope

- VibeVoice-1.5B (can follow later if Large works)
- VibeVoice-0.5B-Streaming
- training or finetuning
- LoRA application
- NVIDIA-specific optimizations (SageAttention, APEX)
- server deployment

## Reference Files

| Component | Source |
|-----------|--------|
| Model definition | `.references/VibeVoice-ComfyUI/vvembed/modular/modeling_vibevoice.py` |
| Inference loop | `.references/VibeVoice-ComfyUI/vvembed/modular/modeling_vibevoice_inference.py` |
| Config | `.references/VibeVoice-ComfyUI/vvembed/modular/configuration_vibevoice.py` |
| Acoustic tokenizer | `.references/VibeVoice-ComfyUI/vvembed/modular/modular_vibevoice_tokenizer.py` |
| Diffusion head | `.references/VibeVoice-ComfyUI/vvembed/modular/modular_vibevoice_diffusion_head.py` |
| DPM-Solver | `.references/VibeVoice-ComfyUI/vvembed/schedule/dpm_solver.py` |
| Processor | `.references/VibeVoice-ComfyUI/vvembed/processor/vibevoice_processor.py` |
| Tokenizer processor | `.references/VibeVoice-ComfyUI/vvembed/processor/vibevoice_tokenizer_processor.py` |
| Text tokenizer | `.references/VibeVoice-ComfyUI/vvembed/modular/modular_vibevoice_text_tokenizer.py` |
| Streamer | `.references/VibeVoice-ComfyUI/vvembed/modular/streamer.py` |
| Weights | `https://huggingface.co/aoi-ot/VibeVoice-Large` (MIT) |
| Papers | arXiv 2508.19205 (VibeVoice), arXiv 2412.08635 (LatentLM) |
