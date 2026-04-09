# Model Behavior Guides

This directory documents the current runtime behavior of each major model family
in `mlx-speech`.

These guides are intentionally short and operational. They are meant to answer:

- what the model is for
- how to call it
- what defaults the repo uses
- what is currently reliable
- what is still limited or missing

Current guides:

- [Step-Audio-EditX](./step-audio-editx.md)
- [VibeVoice](./vibevoice.md)
- [MOSS-TTSD](./moss-ttsd.md)
- [MossTTSLocal](./moss-local.md)
- [MOSS-SoundEffect](./moss-sound-effect.md)
- [CohereASR](./cohere-asr.md)

Guide scope:

- guides describe runtime behavior plus CLI or module entry points
- not every guide implies a top-level re-export from `mlx_speech.generation`
- tokenizer/runtime support code lives under `src/mlx_speech/models/` even when it
  does not have a dedicated guide here

General rule:

- prefer the model-family guide over scattered assumptions in tests or scripts
- if behavior changes, update the corresponding guide in this directory
