# Granite Speech ASR Design

## Architecture Approach

Granite Speech should land as a normal `mlx-speech` ASR family, not as a wrapper around `mlx-audio`.

Target layout:

```text
src/mlx_speech/models/granite_speech_asr/
  __init__.py
  config.py
  tokenizer.py
  feature_extraction.py
  checkpoint.py
  encoder.py
  projector.py
  language_model.py
  model.py
  processor.py
src/mlx_speech/generation/granite_speech_asr.py
src/mlx_speech/asr/_adapters/granite_speech.py
scripts/generate/granite_speech_asr.py
```

`src/mlx_speech/asr/` remains only dispatch/adaptation. The family runtime owns
checkpoint loading, tokenizer/prompt construction, feature extraction, model
forward, and greedy generation.

## Runtime Pipeline

The implementation follows the verified reference path:

```text
audio
  -> 16 kHz mono waveform
  -> STFT: n_fft=512, win_length=400, hop_length=160
  -> HTK 80-bin log-mel
  -> pair-stack consecutive mel frames into 160-dim encoder frames
  -> CTC Conformer encoder
  -> QFormer projector, 3 learned queries per 15-frame window
  -> replace <|audio|> text embeddings with projected audio embeddings
  -> Granite causal LM prefill
  -> greedy token decode with local KV cache
```

Reference anchors:

- `.references/mlx-audio/mlx_audio/stt/models/granite_speech/granite_speech.py`
  lines 519-571 for feature extraction and audio-token count.
- Same file lines 205-231 for the encoder and midpoint self-conditioned CTC.
- Same file lines 384-418 for the QFormer projector.
- Same file lines 573-614 for prompt construction and audio embedding replacement.
- Same file lines 421-473 and 655-694 for LM forward and generation boundaries.

## Boundary Decisions

- No `mlx_lm`: implement Granite LM locally with this checkpoint's fields:
  `embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, and
  `logits_scaling`.
- No `transformers`: use the existing `tokenizers` dependency with
  `tokenizer.json`, `added_tokens.json`, and `chat_template.jinja`.
- No `mlx-audio`: port only the algorithms needed for Granite Speech.
- No public remote alias until a release repo exists or a later decision points
  one at IBM's original BF16 repo. The first support path is local directory
  loading from `models/ibm/granite_4_0_1b_speech/original/`.
- The runtime generation module returns structured text results only. File
  output belongs in CLI/script layers so ASR inference does not mutate local
  artifacts during normal library calls.

## Generation Diagnostics And Outputs

Use a Granite-specific output tree for local transcript artifacts:

```text
outputs/granite_speech_asr/
  transcripts/
  logs/
  summary.json
```

The batch diagnostic script should accept explicit `--audio` paths and a small
default local sample set. Reasonable default inputs are existing WAV files in:

- `outputs/smoke/generated/`
- `outputs/dramabox/`
- `outputs/clone_eval/manual/`
- `outputs/source/`

The script must not modify, copy, or delete these source WAV files. Transcript
paths should preserve enough of the source-relative path under
`outputs/granite_speech_asr/transcripts/` to avoid basename collisions. For
example, `outputs/smoke/generated/vibevoice_smoke.wav` can map to
`outputs/granite_speech_asr/transcripts/smoke/generated/vibevoice_smoke.txt`.

This output path is for human quality inspection and runtime proof. It is not a
durable benchmark dataset, and exact text should not gate the plan. The useful
gate is that selected files transcribe without forbidden dependencies, produce
non-empty text, and record per-file success or failure in `summary.json`.

## Checkpoint Loading

Use the existing `src/mlx_speech/checkpoints/sharded.py` loader for sharded
`.safetensors`.

Original-weight sanitizer:

- Drop `*.num_batches_tracked`.
- Transpose original 1D convolution weights for `up_conv`, `down_conv`, and
  `depth_conv` when tensors are not already converted.
- Preserve explicit alignment reporting before strict load.

Downloaded original checkpoint facts:

- 3 shards, 954 keys.
- `language_model`: 363 BF16 keys.
- `encoder`: 518 BF16 keys plus 16 I64 bookkeeping keys.
- `projector`: 57 BF16 keys.

## Memory And Bounds

The target machine has enough unified memory for the BF16 model, but runtime
must still avoid unbounded allocations:

- Preflight audio length before feature extraction and compute the projected
  audio-token count before building prompt embeddings.
- Enforce the Granite context limit from config (`max_position_embeddings`,
  4096 for the downloaded model). `prompt_tokens + max_new_tokens` must fit or
  fail with a clear `ValueError`.
- Keep generation incremental: retain KV cache and generated token ids, not
  per-step logits or hidden-state history.
- Do not retain the full checkpoint state dict after model weights are loaded.
- Evaluate and release large intermediate groups at stage boundaries where the
  code materializes model weights, audio features, prompt embeddings, and cache.

## Verification Strategy

Unit tests prove each boundary with tiny configs before full-weight tests:

- config parsing from upstream `config.json`
- tokenizer special IDs and prompt rendering
- feature extraction shape and audio-token count
- checkpoint sanitizer and index/header inspection
- encoder/projector tiny forwards
- Granite LM tiny prefill/decode
- memory/context preflight and duplicate-state retention checks
- runtime orchestration with fake modules
- diagnostic output path construction and summary writing without touching
  existing `outputs/smoke/` artifacts

Checkpoint and runtime tests are gated on local model files and skip cleanly
when absent. Full docs/listing updates happen only after local runtime smoke
passes.
