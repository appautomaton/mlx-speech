# Models

This directory is for local model checkpoints and weights used during
development.

Rules:

- Do not put model weights under `src/`.
- Do not package model weights into the library.
- Treat this directory as local working data, not published package contents.
- Prefer local paths and `.safetensors` files when possible.
- Keep original upstream weights separate from MLX-converted weights.

Illustrative current layout:

```text
models/
  openmoss/
    moss_tts_local/
      original/    # Upstream Hugging Face files
      mlx-int8/    # Local MLX-converted 8-bit weights
    moss_audio_tokenizer/
      original/    # Upstream Hugging Face files
      mlx-int8/    # Local MLX-converted 8-bit weights
    moss_ttsd/
      original/
      mlx-int8/
    moss_sound_effect/
      original/
      mlx-4bit/
  vibevoice/
    original/
    mlx-int8/
  cohere/
    cohere_transcribe/
      original/
      mlx-int8/
  stepfun/
    step_audio_editx/
      original/
      mlx-int8/
    step_audio_tokenizer/
      original/
```

This file is kept in git so the directory exists in the repository. Weight
files inside `models/` are ignored by default.
