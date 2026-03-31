# v0: MossTTSLocal — MLX-Native Implementation

## Scope

Build a working `MossTTSLocal` inference path: text in, waveform out.

This is the only v0 target. Do not broaden to `MOSS-TTSD`,
`MOSS-VoiceGenerator`, `MOSS-SoundEffect`, or `MOSS-TTS-Realtime`.

## Pipeline

```
text
  → processor (tokenize, build conversation format)
  → MossTTSLocal (global transformer → local transformer → sample RVQ tokens)
  → Cat audio tokenizer (decode discrete tokens → waveform)
  → 24 kHz audio output
```

The audio tokenizer is part of this plan. Token-only output does not count.

## Current Status

- Stage 1 is implemented and validated against the real upstream shard index.
- Stage 2 and Stage 3 are implemented far enough to strict-load the real
  upstream checkpoint and run a minimal real-weight forward path.
- Stage 3.5 is implemented for both `MossTTSLocal` and `MOSS-Audio-Tokenizer`:
  offline conversion now writes real `mlx-int8/` artifacts, and runtime
  loading from converted weights is validated.
- Stage 4 is implemented for the current scope: the Cat codec encode/decode
  path runs in MLX, and the default runtime now loads the regenerated
  `mlx-int8/` codec artifact directly for both encode and decode.
- Stage 6 is implemented for the current scope: a pure-MLX non-streaming
  generation loop produces `(T, 1+Nq)` rows from the converted speech model.
- KV cache is now the default inference path for batch-size-1 sampled
  generation: the runtime prefills the global Qwen3 backbone once and decodes
  subsequent rows through paired global/local per-layer KV caches.
- The generation hot loop no longer grows outer decode state or inner local
  RVQ inputs through repeated `mx.concatenate(...)`. Both paths now use
  preallocated decode buffers.
- The local full-sequence depth path now restores causal masking, so the
  default uncached runtime no longer lets earlier RVQ depths attend to future
  positions within the same frame.
- Stage 7 is implemented for the runtime path: direct generation, clone,
  continuation, and continuation+clone now run end-to-end on local checkpoints
  through the pure-MLX runtime and CLI.
- Stage 5 is implemented beyond text-only packing: the processor now supports
  generation and continuation conversations, path-based reference audio
  ingestion, MLX audio-code decode helpers, and upstream parity tests for the
  four main inference conversation shapes.
- Real upstream checkpoints for `MOSS-TTS-Local-Transformer` and
  `MOSS-Audio-Tokenizer` are downloaded under `models/openmoss/.../original/`.
- The global Qwen3 RoPE path now matches upstream, and the first generated
  frame under greedy decoding matches upstream on original weights.

## Execution Discipline

- Work one unfinished slice at a time.
- After each completed slice, run the smallest direct validation that proves
  it works.
- Update this file immediately after validation before moving to the next
  unfinished slice.
- Keep runtime code, conversion scripts, tests, and local model artifacts in
  their existing boundaries.

## Completed So Far

- Project scaffold is in place: `uv` package metadata, `src/mlx_speech/`,
  focused tests, and local `models/` layout for original vs converted weights.
- Sharded upstream checkpoint loading is implemented for `.safetensors` plus
  `model.safetensors.index.json`.
- `MossTTSLocalConfig` loads from the real upstream config and reflects the
  true checkpoint values now used by the codebase.
- The MLX `MossTTSLocalModel` module tree matches the upstream checkpoint key
  space exactly for the currently implemented global/local path.
- Strict checkpoint validation passes with no missing keys, no unexpected keys,
  and no shape mismatches for the implemented module tree.
- A minimal real-weight forward path already runs through global transformer,
  global→local adapter, local transformer, and LM heads.
- `scripts/convert_moss_local.py` converts the real upstream speech checkpoint
  into `models/openmoss/moss_tts_local/mlx-int8/`.
- `scripts/convert_moss_audio_tokenizer.py` converts the real upstream codec
  checkpoint into `models/openmoss/moss_audio_tokenizer/mlx-int8/`.
- Converted `mlx-int8/` weights now strict-load through the default runtime
  loader, and the processor can read tokenizer assets from the converted
  directory.
- The Cat codec decode path is implemented in MLX, strict-loads from
  `models/openmoss/moss_audio_tokenizer/mlx-int8/`, and passes a minimal real
  decode forward path.
- The processor path is implemented for prompt formatting, tokenization,
  packing, attention-mask creation, reference-audio ingestion, and audio-code
  decode helpers.
- The generation loop is implemented in `src/mlx_speech/generation/` and drives
  the converted speech model without PyTorch.
- Explicit KV cache types exist under `src/mlx_speech/models/moss_local/`
  for both the global time-axis decode and the local RVQ depth decode.
- `Qwen3Model` / `MosiTTSModel` expose `prefill(...)` and `decode_step(...)`
  APIs used by the default cached path.
- The default runtime now enables both the global time-axis KV cache and the
  local RVQ-depth KV cache for single-item sampled inference, with
  `--no-kv-cache` retained as the explicit uncached fallback path.
- `use_local_kv_cache` is no longer a public config or CLI knob. The local
  RVQ-depth cache is part of the cached runtime itself rather than a separate
  user-facing decision.
- Real end-to-end synthesis now writes WAV output via `scripts/generate_moss_local.py`.
- Generation defaults are now aligned with upstream semantics: separate text
  and audio sampling policies, full `n_vq` decode by default, and natural stop
  on `audio_end_token_id`.
- Output-side listening helpers still exist, but the generation script no
  longer applies trim or peak normalization by default; raw WAV export is now
  the default behavior.
- Text-only synthesis now defaults to the upstream model's own generation
  settings for audio sampling (`temperature=1.0`, `top_p=0.95`, `top_k=50`,
  `repetition_penalty=1.1`) instead of the earlier demo-tuned override.
- Text-only synthesis now auto-estimates `tokens` from prompt length unless
  explicitly overridden, which improves stop behavior for short prompts.
- The global transformer now uses a Qwen3-compatible RoPE implementation
  instead of `mlx.nn.RoPE`, which removed the first verified global-model
  parity break.
- Greedy generation now skips sampling warpers entirely when
  `do_sample=False`, matching upstream behavior.
- The original-weight preview was regenerated after the parity fix to replace
  the older broken output.
- The MLX `MossTTSLocalProcessor` now applies the upstream chat-template
  generation prompt before tokenization, and its `input_ids` /
  `attention_mask` exactly match the upstream processor for direct generation.
- A dedicated `Python 3.12 + transformers 5.0.0` debug environment now exists
  in `.venv`, with the previous `py313` environment moved aside. This was
  used to run a true upstream baseline for `MossTTSLocal`.
- A true upstream baseline output now exists at
  `outputs/moss_local_upstream_true.wav`, generated with upstream processor,
  upstream speech model, upstream codec, local original weights, and the
  upstream-recommended runtime stack.
- The MLX Cat codec decode path now uses an upstream-matching pairwise RoPE
  implementation. On the same upstream-generated codes, MLX codec decode now
  matches upstream codec decode closely (`rmse ~= 9e-4`, `max_abs < 0.01`).
- The MLX Cat codec now also supports encode and `batch_encode`, including
  local audio preprocessing for mono fold-down, resampling to 24 kHz, and
  loudness normalization before tokenization.
- The codec checkpoint loader now keeps encoder weights instead of dropping
  them, and `scripts/convert_moss_audio_tokenizer.py` is aligned with the
  full encoder+decoder model shape.
- Runtime checkpoint loading now preserves stored tensor dtypes. Default
  runtime loading remains `mlx-int8/`, so quantized MLX weights stay on the
  fast path instead of widening at load time.
- The codec `mlx-int8/` artifact has been regenerated with encoder weights
  included. Default codec loading no longer falls back to `original/` during
  reference-audio encode.
- The default quantized runtime now keeps the main activation path in
  `bfloat16` while preserving `float32` for numerically sensitive reductions
  such as RMSNorm accumulation, attention-score math, and LFQ distance
  calculations.
- This mixed-precision policy is now validated on the real quantized runtime:
  speech hidden states and codec encoder hidden states stay in `bfloat16`,
  while final decoded waveform output remains `float32`.
- `MossTTSLocalProcessor` now supports direct generation, clone,
  continuation, and continuation+clone packing. It also exposes
  `encode_audios_from_wav`, `encode_audios_from_path`, `decode_audio_codes`,
  and `decode_sequences`.
- A new conversation-oriented inference surface now exists alongside the
  original text-only helper. Direct TTS remains on
  `synthesize_moss_tts_local(...)`, while richer inference modes flow through
  `synthesize_moss_tts_local_conversations(...)`.
- `scripts/generate_moss_local.py` now supports
  `--mode {generation,clone,continuation,continue_clone}`,
  `--reference-audio`, and the conditioning fields already present in the
  prompt schema.
- The CLI and generation config now support running without a user-visible
  `max_new_tokens` limit while still keeping an internal safety cap, so longer
  samples can stop on EOS without forcing a guessed row budget.
- MLX now distinguishes between upstream model defaults and upstream app
  defaults: the low-level generation config keeps the model's internal
  `generate()` defaults, while the CLI and convenience helper align with the
  `moss_tts_app.py` user-facing defaults and keep duration control off unless
  explicitly requested.
- A dedicated parity script now lives at
  `scripts/compare_moss_local_upstream_parity.py`. Current verified results:
  processor `input_ids`/`attention_mask` parity is exact, original-weight
  greedy rollout matches upstream for the first 17 rows, and same-code codec
  decode matches upstream closely (`rmse ~= 1.1e-3`).
- Current validation now includes:
  - py313 test suite: `56 passed, 4 skipped`
  - `ruff check src scripts tests`
  - py312 upstream parity: processor parity tests pass for direct generation,
    clone, continuation, and continuation+clone
  - CLI smoke runs pass for direct generation, clone, continuation, and
    continuation+clone on original weights
  - KV cache validation:
    - cache append/update tests for global/local cache objects
    - sampled batch-size-1 cache path parity for generation, clone,
      continuation, and continuation+clone
    - batch-size>1 fallback remains on the uncached path
    - cached runtime is now exposed through a single public `use_kv_cache`
      switch, with local RVQ-depth caching kept internal
  - Quantized benchmark (cached runtime vs uncached):
    - cached runtime (global + local KV cache): `5.94s`
    - uncached runtime: `13.77s`
    - speedup: `~2.32x`
    - real-time factor: `~1.64x` faster than real time on the cached path

## Immediate Next Work

1. Extend greedy rollout parity beyond the current 17-row validated window.
2. Quality pass: tighten stop behavior on longer prompts, validate duration
   conditioning stability, and establish known-good generation presets.
3. Release preparation: finalize HuggingFace artifact publication workflow,
   update README and examples to reflect the quantized default runtime.

## Ground Truth From Real Checkpoints

These values are taken from the real upstream `config.json` and checkpoint
files currently stored under `models/openmoss/moss_tts_local/original/`.

**Global Transformer** — Qwen3 backbone used by `model.language_model`:
- Hidden size `2048`
- Intermediate size `6144`
- Number of layers `28`
- Attention heads `16`
- Key/value heads `8`
- Head dim `128`
- Vocabulary size `155648`
- Max position embeddings `40960`
- Uses RoPE in the global transformer

**Local Transformer** — `local_transformer`:
- Hidden size `1536`
- FFN hidden size `8960`
- Number of layers `4`
- Attention still uses head dim `128`
- Real weight shapes show:
  - `q_proj.weight`: `(2048, 1536)`
  - `k_proj.weight`: `(1024, 1536)`
  - `v_proj.weight`: `(1024, 1536)`
  - `o_proj.weight`: `(1536, 2048)`
- Local transformer removes rotary position embeddings

**Adapters**:
- `speech_embedding_to_local_mlp`: `2048 → 2048 → 1536`
- `local_to_speech_embedding_mlps`: `1536 → 2048 → 2048`

**Embeddings and Heads**:
- Text embedding: `Embedding(155648, 2048)`
- Audio embeddings: `32 × Embedding(1025, 2048)`
- Head 0: `2048 → 155648`
- Heads 1–32: `2048 → 1025`

**Cat Audio Tokenizer**:
- Separate checkpoint from `OpenMOSS-Team/MOSS-Audio-Tokenizer`
- Stored locally under `models/openmoss/moss_audio_tokenizer/original/`
- For the current runtime we now support both encode and decode paths

## Architecture Summary

MossTTSLocal is a **hierarchical causal model** with two transformers:

**Global Transformer** — temporal backbone based on Qwen3:
- Hidden size 2048, 28 layers, text vocab 155648
- Encodes linguistic context across time steps
- Uses grouped-query attention with RoPE

**Local Transformer** — depth-wise RVQ predictor:
- 4 layers, hidden size 1536, FFN hidden 8960
- Generates 32 RVQ codebook tokens per time step (coarse-to-fine)
- Pure causal self-attention across codebook depths
- No rotary position embedding

**Adapters** connecting the two:
- `speech_embedding_to_local_mlp`: 2048 → 1536 (gate/up/down, FFN hidden 2048)
- `local_to_speech_embedding_mlps`: 33 MLPs, each 1536 → 2048

**Embeddings:**
- Text: `Embedding(155648, 2048, pad=151643)`
- Audio: 32 × `Embedding(1025, 2048, pad=1024)` plus one text embedding

**Heads:**
- 33 prediction heads with RMSNorm
- Head 0: text logits (2048 → 155648)
- Heads 1–32: audio logits (2048 → 1025)

**Cat Audio Tokenizer** — 1.6B causal transformer codec:
- Input: 24 kHz waveform
- Output frame rate: 12.5 Hz (80ms frames)
- 32-layer RVQ, each 10-bit codebook (vocab 1024)
- Current runtime supports both **encode** and **decode** paths. Decode is on
  the direct text-to-waveform path; encode powers clone and continuation.

## Generation Flow

Per time step:
1. Global transformer produces hidden state `g_t`
2. Adapter MLP maps `g_t` (2048) → local input (1536)
3. Local transformer autoregressively generates RVQ tokens:
   - For each codebook layer k = 0..K:
     - Accumulate: `[g_t, emb(tok_1), ..., emb(tok_k)]`
     - Local transformer forward → logits via `lm_head[k]`
     - Sample with temperature / top-p / top-k / repetition penalty
4. Embed generated tokens back into global space via `local_to_speech_embedding_mlps`
5. Repeat until EOS or max tokens

## Checkpoint Details

**Source:** `OpenMOSS-Team/MOSS-TTS-Local-Transformer` on HuggingFace.
Ships as sharded `.safetensors` with `model.safetensors.index.json`.

**Key layout:**

```
model.language_model.embed_tokens.*           # Qwen3 text embeddings
model.language_model.layers.{0..N}.*          # Qwen3 transformer blocks
model.embedding_list.0.*                      # Text embedding
model.embedding_list.{1..32}.*                # Audio embeddings (per RVQ layer)
local_transformer.layers.{0..3}.self_attn.*   # Local attention
local_transformer.layers.{0..3}.mlp.*         # Local FFN
local_transformer.norm.*                      # Local final norm
speech_embedding_to_local_mlp.*               # Global→local adapter
local_to_speech_embedding_mlps.{0..32}.*      # Local→global adapters
layer_norm_before_lm_heads.{0..32}.*          # Pre-head norms
lm_heads.{0..32}.weight                       # Prediction heads
```

**Cat codec:** Separate checkpoint from `OpenMOSS-Team/MOSS-Audio-Tokenizer`.
Also `.safetensors`.

**Remapping notes:**
- Upstream keys use PyTorch naming. Inspect actual key names against MLX model
  definition and build a `sanitize()` method for any necessary remapping.
- Qwen3 and local attention layers will need careful weight remapping against
  MLX conventions.
- Check for `position_ids`, `bias` buffers, or other non-parameter tensors
  that should be skipped during loading.

## Checkpoint Workflow

The library does NOT load upstream weights directly at runtime. The flow is:

```
upstream safetensors (original/)
  → convert: remap keys to MLX module tree
  → quantize: int8 via mlx.nn.quantize()
  → save: MLX-native safetensors + config → mlx-int8/
```

Runtime loads from `mlx-int8/` only. Conversion is a one-time offline step.

**Rules:**
- Load from local paths in `models/` only.
- Runtime loads `mlx-int8/` — not `original/`.
- `.safetensors` only — no `.bin` support.
- No HuggingFace Hub downloads at runtime.
- User runs conversion once after downloading upstream checkpoints.
- The `mlx-int8/` format is what gets published to HuggingFace later.

**Conversion script:** `scripts/convert_moss_local.py`
- Input: `models/openmoss/moss_tts_local/original/`
- Output: `models/openmoss/moss_tts_local/mlx-int8/`
- Must also convert the Cat codec:
  - Input: `models/openmoss/moss_audio_tokenizer/original/`
  - Output: `models/openmoss/moss_audio_tokenizer/mlx-int8/`

## Reference Files

Read these before implementing. Do not guess architecture.

| Component | Upstream Source |
|-----------|---------------|
| Model config | `.references/MOSS-TTS/moss_tts_local/configuration_moss_tts.py` |
| Model definition | `.references/MOSS-TTS/moss_tts_local/modeling_moss_tts.py` |
| Processor | `.references/MOSS-TTS/moss_tts_local/processing_moss_tts.py` |
| Generation utils | `.references/MOSS-TTS/moss_tts_local/inference_utils.py` |
| Audio tokenizer | `.references/MOSS-TTS/moss_audio_tokenizer/` |
| End-to-end CLI | `.references/MOSS-TTS/clis/moss_tts_app.py` |
| MLX model patterns | `.references/mlx-audio/mlx_audio/utils.py` |
| MLX audio I/O | `.references/mlx-audio/mlx_audio/audio_io.py` |
| MLX checkpoint loading | `.references/mlx-audio/mlx_audio/utils.py` (see `base_load_model`) |
| MLX weight remapping | `.references/mlx-audio/mlx_audio/tts/models/kokoro/kokoro.py` (see `sanitize`) |

## Key Constants

| Name | Value | Notes |
|------|-------|-------|
| `hidden_size` | 2048 | Global transformer |
| `intermediate_size` | 6144 | Global FFN |
| `num_hidden_layers` | 28 | Global transformer depth |
| `num_attention_heads` | 16 | Global attention heads |
| `num_key_value_heads` | 8 | Global KV heads |
| `head_dim` | 128 | Shared attention head dim |
| `local_hidden_size` | 1536 | Local transformer |
| `local_ffn_hidden_size` | 8960 | Local FFN |
| `local_num_layers` | 4 | Local transformer depth |
| `vocab_size` | 155648 | Text vocabulary |
| `n_vq` | 32 | RVQ codebook layers |
| `audio_vocab_size` | 1024 | Per-codebook vocabulary |
| `audio_pad_code` | 1024 | Audio padding token |
| `sampling_rate` | 24000 | Output audio Hz |
| `pad_token_id` | 151643 | Text padding |
| `audio_start_token_id` | 151652 | Audio sequence start |
| `audio_end_token_id` | 151653 | Audio sequence end / EOS |
| `audio_user_slot_token_id` | 151654 | User audio placeholder |
| `audio_assistant_gen_slot_token_id` | 151656 | Generation trigger |

## Implementation Stages

Each stage should be independently testable before moving to the next.

### Stage 1 — Config and Checkpoint Loading

- Define `MossTTSLocalConfig` dataclass from upstream config values.
- Load sharded `.safetensors` from `models/` via `mx.load()`.
- Parse `model.safetensors.index.json` for shard mapping.
- Write a `sanitize()` function for any key remapping needed.
- **Test:** load weights, print key count and shapes, verify no missing/unexpected keys.
- **Status:** done

### Stage 2 — Global Transformer

- Port the Qwen3-based global transformer as `mlx.nn.Module`.
- Attention layers with RoPE and grouped-query attention.
- Embedding list: text + 32 audio embeddings.
- **Test:** random input through global forward, verify output shape `(B, T, 2048)`.
- **Status:** done for current scope; real upstream weights strict-load and run

### Stage 3 — Local Transformer + Adapters + Heads

- 4-layer local transformer.
- `speech_embedding_to_local_mlp` (2048 → 1536).
- `local_to_speech_embedding_mlps` (33 × 1536 → 2048).
- 33 prediction heads with RMSNorm.
- **Test:** mock global hidden → local forward → verify logit shapes.
- **Status:** done for current scope; real upstream weights strict-load and run

### Stage 4 — Cat Audio Tokenizer (Encode + Decode)

- Port the Cat codec encode + decode path as MLX modules.
- Load from converted `mlx-int8/` weights (produced by Stage 3.5).
- **Test:** random token tensor → decode → verify waveform shape and sample
  rate; short waveform → encode → verify code shape and lengths.
- **Status:** done for current scope; the default quantized codec now supports
  both encode and decode in the pure-MLX runtime

### Stage 5 — Processor

- Text tokenization using Qwen3 tokenizer (from checkpoint files).
- Conversation format builder (`build_user_message` equivalent).
- Token packing for the `(B, T, 1+32)` input format.
- **Test:** text → processor → verify input_ids shape and special token placement.
- **Status:** done for current scope; processor now supports generation,
  clone, continuation, continuation+clone, reference-audio encode helpers, and
  audio-code decode helpers

### Stage 6 — Generation Loop

- Autoregressive sampling: global step → local depth loop → token emission.
- Temperature, top-p, top-k, repetition penalty.
- EOS detection on `audio_end_token_id`.
- **Test:** loaded model + processor → generate tokens → verify token sequence shape.
- **Status:** done for current scope; pure-MLX non-streaming generation now follows upstream-style default sampling and reaches EOS on the v0 smoke path

### Stage 7 — End-to-End

- Wire: text → processor → model.generate → Cat decode → waveform → WAV file.
- **Test:** produce an actual audio file from a text prompt using real checkpoints.
- **Status:** done for current scope; default local `mlx-int8` checkpoints now
  run end-to-end for direct generation, clone, continuation, and
  continuation+clone on the pure-MLX runtime

### Stage 2.5 — Real Weight Alignment

- Map real upstream checkpoint keys onto the MLX module tree.
- Resolve attention projection layout and any transpose rules.
- Update `sanitize()` with the full key remapping.
- **Test:** load upstream weights into Stage 2/3 modules with no missing/unexpected keys.
- **Status:** done for the current module tree; key set and shapes match exactly, strict load passes

### Stage 3.5 — MLX Conversion + Int8 Quantization

- Build `scripts/convert_moss_local.py`:
  - Load upstream `original/` safetensors.
  - Apply `sanitize()` key remapping from Stage 2.5.
  - Quantize eligible layers to int8 via `mlx.nn.quantize()`.
  - Save MLX-native safetensors + config to `mlx-int8/`.
- Same for Cat codec: `scripts/convert_cat_codec.py` (or combined script).
- Update model loading to use `mlx-int8/` as the default path.
- **Test:** convert → load from `mlx-int8/` → verify shapes and dtypes match.
- **Output:** ready-to-use MLX weights in `models/openmoss/.../mlx-int8/`.
  These are the artifacts that will eventually be uploaded to HuggingFace.
- **Status:** done for current scope. Both `MossTTSLocal` and
  `MOSS-Audio-Tokenizer` convert from `original/` to `mlx-int8/`, and both
  converted runtime paths strict-load successfully.

## Done Criteria

v0 is done when:
- `MossTTSLocal` generates intelligible speech from text input.
- Output is a 24 kHz waveform, not just discrete tokens.
- Runs entirely on MLX with no torch dependency.
- Loads from converted int8 MLX weights (`mlx-int8/`), not upstream originals.
- Conversion scripts produce HuggingFace-ready MLX safetensors.
- Has focused tests for each stage.

Current codebase is already beyond the original v0 boundary: clone,
continuation, and continuation+clone are implemented on top of the same pure-
MLX runtime.

## Out of Scope

- Voice cloning / reference audio conditioning
- Streaming inference
- Variable bitrate control
- `.bin` checkpoint support
- HuggingFace Hub integration
- Other MOSS family models
