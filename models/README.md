# Models

This directory is for local model checkpoints and weights used during
development.

Rules:

- Do not put model weights under `src/`.
- Do not package model weights into the library.
- Treat this directory as local working data, not published package contents.
- Prefer local paths and `.safetensors` files when possible.
- Keep original upstream weights separate from MLX-converted weights.

Recommended v0 layout:

```text
models/
  openmoss/
    moss_tts_local/
      original/    # Upstream Hugging Face files
      mlx-int8/    # Local MLX-converted 8-bit weights
    moss_audio_tokenizer/
      original/    # Upstream Hugging Face files
      mlx-int8/    # Local MLX-converted 8-bit weights
```

This file is kept in git so the directory exists in the repository. Weight
files inside `models/` are ignored by default.
