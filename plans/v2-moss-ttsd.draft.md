# v2: MOSS-TTSD — Multi-Speaker Dialogue

## Scope

Port the MOSS-TTSD 8B model: multi-speaker conversational dialogue with
per-speaker voice cloning. Architecture is MossTTSDelay (parallel RVQ via
delay scheduling), fundamentally different from v0/v1's MossTTSLocal.

Do not port MOSS-VoiceGenerator, MOSS-SoundEffect, or MOSS-TTS-Realtime.

## Pipeline

```
script with speaker tags ([S1], [S2], ...)
  + per-speaker reference audio
  → Cat codec encode (per-speaker waveform → RVQ tokens)  [reuse current runtime]
  → processor (normalize text, build multi-speaker conversation,
               apply delay pattern)
  → MossTTSDelay generate (parallel RVQ sampling with delay scheduling)
  → processor decode (de-delay pattern → audio codes)
  → Cat codec decode (tokens → waveform)  [reuse current runtime]
  → 24 kHz multi-speaker dialogue output
```

## Prerequisites

The current `MossTTSLocal` runtime already provides the required base pieces:

- Cat codec encode + decode in pure MLX
- quantized local-path runtime loading (`mlx-int8`, `W8Abf16`)
- clone / continuation / continuation+clone on `MossTTSLocal`
- global + local KV cache (design pattern reusable for TTSD's generation loop)

v2 builds on that existing baseline. It does not depend on a separate future
"first cloning implementation" milestone.

## Architecture: MossTTSDelay

**Key difference from MossTTSLocal:** No local transformer. Single Qwen3
backbone predicts all RVQ codebook heads in parallel via delay scheduling.

### Delay Pattern vs Local Depth-First

| Aspect | MossTTSDelay (v2) | MossTTSLocal (v0/v1) |
|--------|-------------------|---------------------|
| RVQ prediction | Parallel (all heads per forward pass) | Sequential (depth-first loop) |
| Architecture | Single Qwen3 backbone | Global Qwen3 + 4-layer local transformer |
| Forward passes per timestep | 1 | 1 + 32 (local depth loop) |
| Model size | 8B | 1.7B |
| Long-context | Up to 60 minutes | Not optimized for long-form |

### Delay Pattern Scheduling

Instead of predicting codebooks sequentially, delay scheduling staggers them
across time:

```
Time →  t0  t1  t2  t3  t4  t5  t6  ...
VQ 0:   a0  a1  a2  a3  a4  a5  a6  ...
VQ 1:   pad a0  a1  a2  a3  a4  a5  ...
VQ 2:   pad pad a0  a1  a2  a3  a4  ...
...
VQ N:   pad pad pad ... a0  a1  a2  ...
```

At each timestep, the model predicts all VQ heads simultaneously. VQ channel k
is delayed by k positions. The delay pattern is applied during input processing
and reversed during output decoding.

```python
# Apply delay pattern (input processing)
# Input: (T, n_vq) — all VQ channels at same timestep
# Output: (T + n_vq - 1, n_vq) — staggered across time
for i in range(n_vq):
    delayed[i : i + T, i] = codes[:, i]

# De-delay pattern (output decoding)
# Input: (T + n_vq - 1, n_vq) — staggered
# Output: (T, n_vq) — realigned
for i in range(n_vq):
    codes[:, i] = delayed[i : i + T, i]
```

### Model Structure

```python
class MossTTSDelayModel:
    language_model: Qwen3Model           # 8B Qwen3 backbone
    emb_ext: [Embedding(1025, hidden)]   # n_vq audio embeddings
    lm_heads: [                          # n_vq + 1 prediction heads
        Linear(hidden, vocab_size),      # Head 0: text
        Linear(hidden, 1025),            # Heads 1..n_vq: audio
        ...
    ]
```

No local transformer, no adapters, no per-head RMSNorm. The Qwen3 backbone
does all the work. Audio heads mask the padding token (index 1024) to `-inf`
during logit computation.

### Embedding Computation

```
inputs_embeds = language_model.embed_tokens(input_ids[..., 0])   # text
for i, embed in enumerate(emb_ext):
    inputs_embeds += embed(input_ids[..., i + 1])                # audio
```

All channels summed into one embedding, same as MossTTSLocal.

## MOSS-TTSD Specific: Multi-Speaker Dialogue

### Speaker Representation

- Text uses speaker tags: `[S1]`, `[S2]`, ..., `[S5]`
- Up to 5 speakers per dialogue session
- Each speaker has:
  - Reference audio (`prompt_audio_speakerN`) for voice cloning
  - Prompt text (`prompt_text_speakerN`) for identity establishment

### Text Normalization

TTSD normalizes dialogue text before processing:
- `[1]` → `[S1]` (numeric to speaker tag)
- Remove special punctuation (brackets, quotes, dashes)
- `哈哈哈` → `[笑]`, `ha ha ha` → `[laugh]`
- Replace long dashes/ellipses with commas
- Collapse duplicate punctuation
- Merge consecutive same-speaker tags

### Multi-Speaker Generation Flow

```python
# Per-speaker reference audio encoding (reuse current Cat codec runtime)
references = []
for speaker_id in speaker_ids:
    wav = load_mono_wav(audio_map[speaker_id])
    wav = resample_to_24k(wav)
    codes = cat_codec.encode(wav)  # (T, n_vq)
    references.append(codes)

# Build prefixed text with speaker tags
text = "[S1]Hello! [S2]Hi there! [S1]How are you?"

# Build conversation
user_msg = processor.build_user_message(
    text=text,
    reference=references,  # per-speaker
)

# Generate
outputs = model.generate(
    input_ids=processor([user_msg], mode="generation"),
    max_new_tokens=8192,
    audio_temperature=1.1,
    audio_top_p=0.9,
    audio_repetition_penalty=1.1,
)
```

## Configuration

From upstream `configuration_moss_tts.py` (MossTTSDelayConfig):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_type` | `moss_tts_delay` | |
| `n_vq` | 16 | TTSD uses 16 codebooks (not 32) |
| `audio_vocab_size` | 1024 | Per-codebook vocabulary |
| `audio_pad_code` | 1024 | Padding token |
| `sampling_rate` | 24000 | |
| `pad_token_id` | 151643 | |
| `im_start_token_id` | 151644 | |
| `im_end_token_id` | 151645 | |
| `audio_start_token_id` | 151652 | |
| `audio_end_token_id` | 151653 | |
| `audio_user_slot_token_id` | 151654 | |
| `audio_assistant_gen_slot_token_id` | 151656 | |
| `audio_assistant_delay_slot_token_id` | 151662 | Delay pattern offset marker |
| `language_config` | Qwen3Config | 8B backbone |

**Note:** The exact Qwen3 config values (hidden_size, num_layers, etc.) must
be read from the real MOSS-TTSD checkpoint's `config.json` once downloaded.
The upstream TTSD repo references `OpenMOSS-Team/MOSS-TTSD-8B` but the
checkpoint must be verified before pinning architecture values.

## Generation Loop: Delay Pattern Tracking

The generation loop tracks per-batch state:

- `audio_lengths`: how many VQ tokens generated for current audio segment
- `delayed_lengths`: position in delay offset (0 to n_vq)
- `is_audio`: whether currently in audio generation mode
- `is_stopping`: whether batch item has finished

Per timestep:
1. Forward pass → logits for all `n_vq + 1` heads
2. **Text head sampling:**
   - If `delayed_lengths < n_vq`: emit `audio_assistant_delay_slot_token_id`
   - If `delayed_lengths == n_vq`: emit `audio_end_token_id` (audio EOS)
   - Otherwise: sample from text logits
   - Text logits mask out special tokens based on `is_audio` state
3. **Audio head sampling (parallel):**
   - `pre_audio_mask`: `audio_lengths > channel_index`
   - `post_audio_mask`: `channel_index > delayed_lengths - 1`
   - `sampling_mask = pre_audio_mask & post_audio_mask`
   - Only sample heads where mask is True; others get `audio_pad_code`
   - Channel 0 (coarsest): separate sampling with repetition penalty
   - Channels 1+: batch-sampled together
4. Update `audio_lengths`, `delayed_lengths` based on emitted tokens
5. Concatenate new token to generation_ids, extend attention mask
6. Stop when all batch items have emitted `im_end_token_id`

## Implementation Stages

### Stage 1 — Config and Checkpoint Loading

- Define `MossTTSDelayConfig` dataclass (simpler than MossTTSLocal — no local
  transformer config fields)
- Download MOSS-TTSD checkpoint
- Parse sharded safetensors via existing `sharded.py` infrastructure
- Map checkpoint keys to MLX module tree
- Write `sanitize()` for key remapping
- **Test:** load weights, verify no missing/unexpected keys

### Stage 2 — MossTTSDelay Model (MLX)

- Single Qwen3 backbone (reuse `Qwen3Model` from v0)
- `emb_ext`: `n_vq × Embedding(1025, hidden_size)` audio embeddings
- `lm_heads`: `(n_vq + 1)` linear heads
- `_compute_input_embeddings`: sum text + all audio channel embeddings
- No local transformer, no adapters, no per-head norms
- **Test:** load real weights → random input → verify output logits shapes

### Stage 3 — MLX Conversion + Int8

- Extend conversion scripts for the 8B Delay model
- Input: `models/openmoss/moss_ttsd/original/`
- Output: `models/openmoss/moss_ttsd/mlx-int8/`
- Same workflow: remap keys → quantize → save MLX safetensors
- 8B model benefits more from int8 than the 1.7B Local model
- **Test:** convert → load from mlx-int8 → verify shapes and dtypes

### Stage 4 — Delay Pattern Processor

- `MossTTSDelayProcessor`: extends the base processor pattern
- Add `apply_delay_pattern(codes, pad_code)`: `(T, n_vq) → (T + n_vq - 1, n_vq)`
- Add `apply_de_delay_pattern(delay_codes)`: reverse
- Multi-speaker text normalization (`normalize_text` from generation_utils)
- Speaker field collection and reference encoding (reuse current Cat encode)
- **Test:** apply delay → de-delay → verify round-trip identity

### Stage 5 — Generation Loop

- Implement the delay-aware autoregressive loop described above
- State tracking: `audio_lengths`, `delayed_lengths`, `is_audio`, `is_stopping`
- Text head: special token masking + sampling
- Audio heads: parallel masked sampling with delay constraints
- Repetition penalty per VQ channel
- KV cache support for efficient generation
- **Test:** generate from loaded model → verify token sequence structure

### Stage 6 — Multi-Speaker Pipeline

- Per-speaker reference audio encoding via v1's Cat encode
- Speaker tag normalization and text preprocessing
- `prepare_sample()` that builds conversation from speaker fields
- Batch inference support (`run_infer_batch`)
- Output: per-segment WAV files merged into final dialogue audio
- **Test:** multi-speaker dialogue script → WAV output with distinct voices

### Stage 7 — End-to-End + CLI

- Wire: script → processor → model.generate → de-delay → Cat decode → WAV
- CLI:
  ```
  mlx-voice dialogue \
    --script dialogue.jsonl \
    --speaker1-audio s1.wav --speaker1-text "[S1]大家好" \
    --speaker2-audio s2.wav --speaker2-text "[S2]你好" \
    -o output.wav
  ```
- JSONL batch mode for processing multiple dialogues
- **Test:** produce multi-speaker dialogue WAV from real checkpoints

## Reference Files

| Component | Upstream Source |
|-----------|---------------|
| Delay model config | `.references/MOSS-TTS/moss_tts_delay/configuration_moss_tts.py` |
| Delay model definition | `.references/MOSS-TTS/moss_tts_delay/modeling_moss_tts.py` |
| Delay processor | `.references/MOSS-TTS/moss_tts_delay/processing_moss_tts_delay_with_codec.py` |
| Delay processor (shared) | `.references/MOSS-TTS/moss_tts_delay/processing_moss_tts.py` |
| TTSD generation utils | `.references/MOSS-TTSD/generation_utils.py` |
| TTSD inference | `.references/MOSS-TTSD/inference.py` |
| TTSD gradio demo | `.references/MOSS-TTSD/gradio_demo.py` |
| Inference sampling | `.references/MOSS-TTS/moss_tts_delay/inference_utils.py` |
| Cat codec | Reuse current MLX Cat codec runtime |
| Qwen3 backbone | Reuse current MLX `Qwen3Model` (scale to 8B config) |

## Checkpoint Layout

```
models/openmoss/
  moss_ttsd/
    original/     ← MOSS-TTSD-8B sharded safetensors
    mlx-int8/     ← converted MLX int8 weights
  moss_audio_tokenizer/
    original/     ← (already downloaded)
    mlx-int8/     ← (already converted in v0/v1)
```

## Checkpoint Workflow

```
models/openmoss/moss_ttsd/original/
  → convert: remap Delay model keys to MLX module tree
  → quantize: int8 via mlx.nn.quantize()
  → save: MLX-native safetensors + config → mlx-int8/
```

Cat codec weights reused from v0/v1 conversion — no reconversion needed.

## Done Criteria

v2 is done when:
- MOSS-TTSD generates multi-speaker dialogue from script input
- Per-speaker voice cloning works via reference audio
- Output is a 24 kHz waveform with distinct speaker voices
- Runs entirely on MLX with no torch dependency
- Loads from converted int8 MLX weights (`mlx-int8/`)
- Supports 1–5 speakers per dialogue
- CLI provides `dialogue` command with JSONL batch support

## Out of Scope

- Streaming inference
- MOSS-VoiceGenerator
- MOSS-SoundEffect
- MOSS-TTS-Realtime
- Fine-tuning / training
- 60-minute sessions (validate shorter durations first)
