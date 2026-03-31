# v3: Cohere Transcribe — MLX ASR

## Scope

Port CohereLabs/cohere-transcribe-03-2026 to pure MLX. This is the first ASR
family in mlx-voice — a fundamentally different task from TTS: audio in,
text out.

The model is a 2B Conformer encoder-decoder. The encoder (ParakeetEncoder,
Fast-Conformer style, 48 layers) is the architectural centerpiece and the
primary porting effort. The decoder is a compact Transformer (8 layers) with
fixed positional embeddings and cross-attention to the encoder.

Reference implementation lives entirely in HuggingFace transformers:
`.references/transformers/src/transformers/models/cohere_asr/` and
`.references/transformers/src/transformers/models/parakeet/` (to be added).

## Architecture

### Full Model Tree

```
CohereAsrForConditionalGeneration
├── model
│   ├── encoder: ParakeetEncoder               ← Conformer, 48 layers, h=1280
│   └── decoder: CohereAsrDecoder              ← Transformer, 8 layers, h=1024
│       ├── embed_tokens: Embedding(16384, 1024)
│       ├── pos_emb: Embedding(1024, 1024)     ← fixed sinusoidal table
│       ├── embedding_layernorm: LayerNorm(1024)
│       ├── proj: Linear(1280→1024, bias=True) ← encoder hidden projection
│       ├── layers × 8: CohereAsrDecoderLayer
│       │   ├── input_layernorm: LayerNorm(1024)
│       │   ├── self_attn: CohereAsrSelfAttention (8h, causal, no RoPE)
│       │   ├── post_attention_layernorm: LayerNorm(1024)
│       │   ├── encoder_attn: CohereAsrCrossAttention (8h, non-causal)
│       │   ├── final_layernorm: LayerNorm(1024)
│       │   └── mlp: Linear(1024→4096) + relu + Linear(4096→1024)
│       └── norm: LayerNorm(1024)
└── proj_out: Linear(1024→16384, bias=True)    ← weight-tied to embed_tokens
```

### Key Configuration (decoder)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 16384 |
| `hidden_size` | 1024 |
| `num_hidden_layers` | 8 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 8 (no GQA) |
| `intermediate_size` | 4096 |
| `hidden_act` | relu |
| `max_position_embeddings` | 1024 |
| `attention_bias` | True |
| `pad_token_id` | 2 |
| `eos_token_id` | 3 |

### Key Configuration (encoder / ParakeetEncoder)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 1280 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 8 |
| `intermediate_size` | 5120 |
| `hidden_act` | silu |
| `conv_kernel_size` | 9 |
| `subsampling_factor` | 8 |
| `subsampling_conv_channels` | 256 |
| `num_mel_bins` | 128 |
| `subsampling_conv_kernel_size` | 3 |
| `subsampling_conv_stride` | 2 |
| `max_position_embeddings` | 5000 |
| `attention_bias` | True |

### Feature Extraction Parameters

| Parameter | Value |
|-----------|-------|
| `sampling_rate` | 16000 Hz |
| `n_fft` | 512 |
| `hop_length` | 160 |
| `win_length` | 400 |
| `num_mel_bins` | 128 |
| `fmin` | 0.0 |
| `fmax` | 8000 Hz |
| `mel_norm` | slaney |
| `preemphasis` | 0.97 |
| `dither` | 1e-5 (seeded by waveform length) |
| `LOG_ZERO_GUARD_VALUE` | 2^-24 |
| `max_audio_clip_s` | 35.0 s |
| `chunk_overlap_s` | 5.0 s (boundary search window) |

### Weights

Single file: `model.safetensors` (4.13 GB BF16, ~2 GB int8 after conversion).
No shards.

### Tokenizer

SentencePiece BPE, 16k vocab, byte fallback. Files: `tokenizer.json`,
`tokenizer.model`. Language and punctuation are communicated via decoder prompt
tokens prepended to the decoder input sequence at generation time.

## What Is New vs Existing Library Code

| Component | Can reuse | New work |
|-----------|-----------|----------|
| Checkpoint loading from safetensors | yes (pattern) | weight remapping for new key layout |
| Int8 quantization | yes (pattern) | apply to encoder + decoder linears |
| Audio I/O, resampling | yes (`audio/`) | — |
| Log-Mel STFT pipeline | no | numpy STFT + mel filterbank |
| Tokenizer wrapper | yes (pattern) | SentencePiece model, prompt builder |
| ParakeetEncoder (Conformer) | no | full new implementation |
| CohereAsrDecoder | no | standard Transformer + cross-attn |
| Greedy decoder generation | no (TTS loops are autoregressive text→audio) | audio→text greedy loop |

## Implementation Stages

> **Status:** All stages (0–8) complete.
> - Exact-match weight loading from both original NeMo and converted mlx-int8 checkpoints
> - Encoder + decoder forward pass verified
> - Greedy transcription loop working end-to-end
> - mlx-int8 checkpoint at `models/cohere/cohere_transcribe/mlx-int8/`

### Stage 0 — Weight Layout Inspection ✓

The Parakeet encoder reference is already read and documented below. The
remaining gating task before Stage 4 is inspecting the actual weight keys.

- Download `model.safetensors` to `models/cohere/cohere_transcribe/original/`
  along with all JSON and tokenizer files
- Run a weight key inspection script to enumerate all keys, shapes, dtypes
- Map every key to the planned MLX module tree (see Stage 4 for the tree);
  write the sanitize table
- Record: which keys are tied (`proj_out.weight` ↔ `embed_tokens.weight`),
  which keys exist only in one direction

**Exit criterion:** complete key→module mapping table, no unknown keys.

### Stage 1 — Config

- `src/mlx_voice/models/cohere_asr/config.py`
- `ParakeetEncoderConfig` frozen dataclass matching the encoder sub-config
- `CohereAsrConfig` frozen dataclass: decoder fields + nested encoder config
- `from_dict(payload)`, `from_path(model_dir)`, `to_dict()`
- Validate: load `config.json` from the original checkpoint directory

**Test:** config round-trip (`from_path` → `to_dict` → `from_dict` equals
original).

### Stage 2 — Feature Extraction

Port the log-Mel pipeline to pure numpy (no torch, no librosa at inference).

- `src/mlx_voice/models/cohere_asr/feature_extraction.py`
- Mel filterbank: replicate `librosa.filters.mel(norm="slaney")` in numpy
  exactly — this is the most numerically sensitive step; validate against
  librosa output on known input before proceeding
- `_stft_power(waveform)`: numpy STFT → power spectrum
  (n_fft=512, hop=160, win=400, Hann, center-pad constant)
- `_apply_preemphasis(waveform, coeff=0.97)`
- `_apply_dither(waveform, amount=1e-5, seed_by_length=True)`
- `_log_mel(waveform)`: power → mel → log(x + 2^-24) → (frames, 128)
- `_normalize(features, attention_mask)`: per-sample zero-mean unit-variance
  masked normalization
- `_split_chunks_energy(waveform)`: energy-based split for audio > 30s
- `CohereAsrFeatureExtractor.__call__`: full pipeline from raw waveform

**Test:** extract features from a known 1-second sine wave and a real audio
clip; compare output to the torch reference implementation (run offline once
to capture reference values).

### Stage 3 — Tokenizer

- `src/mlx_voice/models/cohere_asr/tokenizer.py`
- Load `tokenizer.json` (via `tokenizers` library, already a dep) or
  `tokenizer.model` (SentencePiece); confirm which path `tokenizers` needs
- `CohereAsrTokenizer.encode(text: str) -> list[int]`
- `CohereAsrTokenizer.decode(ids: list[int], skip_special=True) -> str`
- `get_decoder_prompt_ids(language: str, punctuation: bool) -> list[int]`
  Supported: ar de el en es fr it ja ko nl pl pt vi zh
- `CohereAsrProcessor`: wraps feature extractor + tokenizer

**Test:** encode → decode round-trip on a known string; verify prompt ids for
`language="en", punctuation=True` match the reference processor output.

### Stage 4 — ParakeetEncoder

Full architecture is known from reading the reference. Implement in MLX:

- `src/mlx_voice/models/cohere_asr/encoder.py`

**Subsampling** (`ParakeetEncoderSubsamplingConv2D`):
  - Operates on the mel spectrogram as a 2D image: `(batch, 1, T_mel, 128)`
  - 3 Conv2d layers (subsampling_factor=8=2^3, one stride-2 per layer):
    - Layer 1: `Conv2d(1, 256, k=3, stride=2, pad=1)` + ReLU
    - Layers 2-3: depthwise `Conv2d(256, 256, k=3, s=2, pad=1, groups=256)` +
      pointwise `Conv2d(256, 256, 1)` + ReLU
  - Reshape: `(batch, T', 256, 16)` → `(batch, T', 4096)` (256 ch × 16 freq)
  - `Linear(4096, 1280, bias=True)` → `(batch, T', 1280)`
  - `scale_input=False` for cohere, so no `sqrt(hidden_size)` scaling

**Relative Positional Encoding** (`ParakeetEncoderRelPositionalEncoding`):
  - Covers `2*T - 1` positions: `torch.arange(T-1, -T, -1)` (largest to most
    negative)
  - Standard `inv_freq = 1 / (10000 ** (2i / d))`
  - Output: interleaved sin+cos, shape `(batch, 2T-1, hidden_size)`

**Attention** (`ParakeetEncoderAttention`):
  - Transformer-XL style relative positional attention (paper 1901.02860)
  - Projections: `q_proj, k_proj, v_proj, o_proj` (all biased, MHA no GQA)
  - `relative_k_proj`: Linear(1280, 1280, no bias) for position key
  - `bias_u`, `bias_v`: learnable `(num_heads, head_dim)` parameter vectors
  - Two-term computation:
    - matrix_ac: `(q + bias_u) @ k.T * scale` — standard content attention
    - matrix_bd: `(q + bias_v) @ rel_k.T * scale` with `_rel_shift` → used as
      additive bias to matrix_ac before softmax
  - `_rel_shift`: pad-left-1 → reshape → slice to remove extra row
  - Non-causal (bidirectional encoder)

**Convolution Module** (`ParakeetEncoderConvolutionModule`):
  - `pointwise_conv1`: Conv1d(1280, 2560, 1) → GLU (→ 1280)
  - `depthwise_conv`: Conv1d(1280, 1280, 9, groups=1280, pad=4)
  - **`BatchNorm1d(1280)`** — not LayerNorm; must be in eval mode at inference
  - `silu` activation
  - `pointwise_conv2`: Conv1d(1280, 1280, 1)

**Feed-Forward** (`ParakeetEncoderFeedForward`):
  - `Linear(1280, 5120)` → silu → `Linear(5120, 1280)`, both biased

**Conformer Block** (`ParakeetEncoderBlock`):
  - Pre-norms on all sub-modules; 5 LayerNorms per block:
    `norm_feed_forward1`, `norm_self_att`, `norm_conv`, `norm_feed_forward2`,
    `norm_out`
  - Residual scaling: FF sub-modules use 0.5 factor, others use 1.0:
    ```
    x = x + 0.5 * FF1(norm_ff1(x))
    x = x + attn(norm_attn(x), pos_emb)
    x = x + conv(norm_conv(x))
    x = x + 0.5 * FF2(norm_ff2(x))
    x = norm_out(x)
    ```

**ParakeetEncoder forward**:
  1. Subsampling → `(batch, T', 1280)`
  2. Compute relative pos embeddings from subsampled hidden states
  3. 48 × ConformerBlock(hidden, attn_mask, pos_emb)
  4. Return `(last_hidden_state, output_attention_mask)`

**Test:** load real encoder weights (BF16, no quantization yet) → pass random
`(1, 100, 128)` mel features → verify output shape is `(1, T', 1280)` where
`T' = ceil(ceil(ceil(100/2)/2)/2) = 13`.

### Stage 5 — Decoder

- `src/mlx_voice/models/cohere_asr/decoder.py`
- `CohereAsrSelfAttention`: standard MHSA, no RoPE, causal mask, biases
- `CohereAsrCrossAttention`: non-causal, q from decoder, k/v from encoder
- `CohereAsrDecoderLayer`: pre-norm self-attn → cross-attn → MLP
- `CohereAsrDecoder`:
  - `embed_tokens` + `pos_emb` + `embedding_layernorm`
  - `proj`: Linear(1280→1024) projects encoder hidden states once
  - 8 × DecoderLayer
  - `norm`: final LayerNorm
- `CohereAsrModel`: wraps encoder + decoder
- `CohereAsrForConditionalGeneration`: adds `proj_out` with weight tie
- Weight loading; verify tied weights loaded correctly

**Test:** load real decoder weights → pass fake encoder states `(1, 62, 1280)`
and `decoder_input_ids=[decoder_start_token_id]` → verify logit shape is
`(1, 1, 16384)`.

### Stage 6 — Checkpoint Loading + Quantization

- `src/mlx_voice/models/cohere_asr/checkpoint.py`
- `load_cohere_asr_checkpoint(model_dir)`: loads safetensors, returns raw
  weight dict
- `sanitize(weights)`: applies the remapping table from Stage 0
- `load_checkpoint_into_model(model, checkpoint, strict=True)`: verifies all
  keys present and shapes match; returns `AlignmentReport`
- `QuantizationConfig` (or reuse existing pattern)
- `quantize_cohere_asr_model(model, config)`: quantize encoder + decoder
  linear layers; leave subsampling conv and small MLP layers in float
  (evaluate quality — likely keep BN/LN layers float)
- `save_cohere_asr_model(model, output_dir, config, quantization)`

**Test:** load int8 quantized weights → strict load → no missing/unexpected
keys → smoke forward pass on random inputs.

### Stage 7 — Greedy Inference Loop

- `src/mlx_voice/generation/cohere_asr.py`
- `CohereAsrGenerationConfig`: `max_new_tokens`, `language`, `punctuation`
- Inference flow:
  1. Feature extraction → `(1, T_mel, 128)` mel features
  2. Encoder forward → `(1, T_enc, 1280)` encoder hidden states (single pass)
  3. Project encoder states via `decoder.proj` once; cache for all decoder steps
  4. Decoder prefill: `decoder_input_ids = [decoder_start_token_id] + prompt_ids`
  5. Autoregressive decode: per-step `decoder_layer(hidden, encoder_states)` →
     logit → greedy argmax → append → stop on EOS (id=3) or max_new_tokens
  6. Decode token ids → text via tokenizer (skip special tokens)
- Long-form: if audio was chunked in Stage 2, transcribe each chunk and
  concatenate decoded text
- `transcribe(audio, config) -> str`

**Test:** real model (int8) → transcribe a 5-second known English clip →
verify the output matches the reference PyTorch transcription character-for-
character on greedy decode.

### Stage 8 — Conversion Script + CLI

- `scripts/convert_cohere_asr.py`
  - Loads original safetensors from `models/cohere/cohere_transcribe/original/`
  - Applies sanitize, loads into model, quantizes, saves to
    `models/cohere/cohere_transcribe/mlx-int8/`
  - Prints key alignment report

- `scripts/transcribe_cohere_asr.py`
  ```
  python scripts/transcribe_cohere_asr.py \
    --audio input.wav \
    --language en \
    [--no-punctuation] \
    [--model-dir models/cohere/cohere_transcribe/mlx-int8/] \
    [-o transcript.txt]
  ```

**Test:** convert script runs cleanly; CLI produces correct transcription on
demo audio.

## Checkpoint Layout

```
models/cohere/
  cohere_transcribe/
    original/           ← model.safetensors (4.13 GB) + JSON + tokenizer files
    mlx-int8/           ← converted MLX int8 weights + config
```

## Dependency Notes

- No new heavy dependencies expected.
- Mel filterbank replication in numpy: verify against librosa once offline,
  then remove librosa from the inference path entirely.
- SentencePiece via `tokenizers` library (already a dep): confirm byte-fallback
  BPE works without the `sentencepiece` package. If `tokenizers` cannot handle
  it, add `sentencepiece` as a minimal dep.
- No `torch` in the inference path.

## Done Criteria

v3 is done when:
- `transcribe_cohere_asr.py` produces correct greedy transcription for at
  least English and one other language
- Output is character-for-character identical to PyTorch greedy decode on
  known short clips
- Conversion script runs cleanly on the original safetensors
- Model loads from mlx-int8 checkpoint with strict weight alignment
- Long-form audio (>35s) transcribes via energy-based chunking
- Runtime is pure MLX, no torch dependency in the inference path

## Out of Scope

- Beam search (greedy is sufficient for bring-up and most use cases)
- Timestamp / word-level alignment (model does not support it)
- Speaker diarization
- Automatic language detection (model requires explicit language)
- Streaming transcription
- Training or fine-tuning
- Any model other than cohere-transcribe-03-2026

## Reference Files

| Component | Source |
|-----------|--------|
| Decoder model | `.references/transformers/src/transformers/models/cohere_asr/modeling_cohere_asr.py` |
| Modular source | `.references/transformers/src/transformers/models/cohere_asr/modular_cohere_asr.py` |
| Feature extraction | `.references/transformers/src/transformers/models/cohere_asr/feature_extraction_cohere_asr.py` |
| Config | `.references/transformers/src/transformers/models/cohere_asr/configuration_cohere_asr.py` |
| Processor | `.references/transformers/src/transformers/models/cohere_asr/processing_cohere_asr.py` |
| Encoder (Parakeet) | `.references/transformers/src/transformers/models/parakeet/modeling_parakeet.py` |
| Weights | `https://huggingface.co/CohereLabs/cohere-transcribe-03-2026` (Apache 2.0) |
| Transformers ref commit | `8213e0d920d52cb00dcade16b6d1f6e952ac0a8c` (2026-03-30) |
