# Model Behavior Guides

This directory documents the current runtime behavior of each major model family
in `mlx-voice`.

These guides are intentionally short and operational. They are meant to answer:

- what the model is for
- how to call it
- what defaults the repo uses
- what is currently reliable
- what is still limited or missing

Current guides:

- [VibeVoice](./vibevoice.md)
- [MOSS-TTSD](./moss-ttsd.md)
- [MossTTSLocal](./moss-local.md)
- [MOSS-SoundEffect](./moss-sound-effect.md)

General rule:

- prefer the model-family guide over scattered assumptions in tests or scripts
- if behavior changes, update the corresponding guide in this directory
