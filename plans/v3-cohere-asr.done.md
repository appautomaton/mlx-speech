# v3: Cohere Transcribe — MLX ASR

**Status: DONE** (2026-03-31)

## Scope

Port CohereLabs/cohere-transcribe-03-2026 to pure MLX for local Apple Silicon
inference. This is the first ASR family in mlx-speech: audio in, text out,
with no torch-backed runtime.

## What Was Delivered

- pure MLX Cohere ASR runtime under `src/mlx_speech/models/cohere_asr/`
- MLX-native feature extraction, tokenizer wrapper, checkpoint loading, and
  generation loop
- local `mlx-int8` checkpoint support at
  `models/cohere/cohere_transcribe/mlx-int8/`
- user-facing transcription entry point in
  `src/mlx_speech/generation/cohere_asr.py`
- CLI conversion and inference scripts:
  - `scripts/convert_cohere_asr.py`
  - `scripts/transcribe_cohere_asr.py`
- inference-focused API surface:
  - `CohereAsrModel.from_dir(...)`
  - `CohereAsrModel.from_path(...)`
  - `transcribe(...)`
  - `transcribe_batch(...)`
  - prompt-level `language`, `punctuation`, and `itn` control

## Final Runtime Shape

```text
waveform
  -> feature extractor
  -> Parakeet encoder
  -> prompted decoder prefill
  -> greedy autoregressive decode
  -> transcript text
```

Delivered model structure:

- Parakeet encoder: 48-layer Fast-Conformer style stack
- decoder: 8-layer Transformer with cross-attention
- SentencePiece BPE tokenizer with decoder prompt tokens
- chunked long-form transcription path for audio beyond the single-window fast
  path

## Bring-up Fixes That Mattered

Two source-faithfulness bugs were the real blockers during bring-up:

- feature front-end parity:
  - mel filterbank math needed to match the Slaney-normalized reference path
- encoder subsampling parity:
  - the flatten order before the final subsampling projection had to preserve
    `(channels, freq)` rather than `(freq, channels)`

Additional parity cleanup that landed on the way:

- feature-length validity fix
- sample-variance normalization fix
- config-driven feature extractor setup
- prompt/tokenizer parity cleanup
- stricter relative-attention and encoder mask handling

## Validation

The Cohere MLX path was verified against the official reference behavior on the
same local checkpoints and audio.

Real end-to-end checks now work on:

- Hank reference audio
- Peggy reference audio
- Sherlock reference clip

Repository validation at completion:

- `uv run pytest` -> `163 passed, 6 skipped`
- `uv run ruff check .` -> passed

## Follow-on Notes

This port is intentionally inference-first. It does not try to reproduce the
full HuggingFace training or model-management surface.

Still intentionally out of the public API for now:

- timestamps output
- diarization output
- full HF-style processor/model wrapper surface
