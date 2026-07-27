# Project

`mlx-speech` is an open-source, MLX-native speech library for Apple Silicon.

## Why it exists

Speech models ship as PyTorch. Running one on a Mac usually means dragging in
torch, a CUDA-shaped inference path, and a framework's worth of transitive
dependencies to produce a waveform on hardware that never needed any of it.

This repo takes the other route. Every model family is reimplemented against MLX
directly, reading upstream source as a specification rather than importing it as
a runtime. The result is a small dependency surface, checkpoints in
`.safetensors`, and inference that runs natively on unified memory.

## What it covers

Two capability families behind one consistent interface.

**Text to speech.** `fish-s2-pro`, `vibevoice`, `longcat`, `moss-local`,
`moss-ttsd`, `moss-sound-effect`, `step-audio`, `dramabox`. Voice cloning,
dialogue, sound effects, and 48 kHz stereo diffusion.

**Speech to text.** `cohere-asr`, `qwen3-asr-1.7b`, and IBM Granite Speech 4.0 1B
from a local checkpoint.

Plus `reuse`, a pure-MLX SEMamba speech enhancer used as an optional pre-encode
denoiser for voice-reference audio.

Converted weights are published under `appautomaton` on Hugging Face.

## The line it holds

Not a framework. Dependencies are added only when an implementation proves one
necessary, and the public API stays small enough to maintain as OSS for years
rather than months. Breadth of model support is the goal. Breadth of abstraction
is not.
