# VibeVoice

## What It Is

`VibeVoice` is the long-form speech model in this repo. It is best thought of
as:

- text-to-speech with optional voice conditioning
- diffusion-based speech decoding
- a research-oriented path for long conversational synthesis

Current local default runtime:

- `models/vibevoice/mlx-int8`

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("vibevoice")
result = model.generate("Hello from VibeVoice!")
# result.waveform, result.sample_rate
```

```bash
mlx-speech tts --model vibevoice --text "Hello!" -o output.wav
```

Local path (skips HF download):

```bash
mlx-speech tts \
  --model models/vibevoice/mlx-int8 \
  --text "Hello!" \
  -o output.wav
```

## Current Python Module API

Current entry points are split across direct modules.

- `mlx_speech.models.vibevoice.checkpoint` for `load_vibevoice_model(...)`
- `mlx_speech.models.vibevoice.tokenizer` for `VibeVoiceTokenizer.from_path(...)`
- `mlx_speech.generation.vibevoice` for `generate_vibevoice(...)`,
  `synthesize_vibevoice(...)`, and generation config/output types

They are not re-exported from `mlx_speech.generation` yet.

Core entry points:

- `load_vibevoice_model(...)`
- `VibeVoiceTokenizer.from_path(...)`
- `synthesize_vibevoice(...)`

Current user-facing inputs:

- `text: str`
- `reference_audio: mx.array | None`
- `voice_samples: list[mx.array] | None`
- `config: VibeVoiceGenerationConfig`

`voice_samples` is the direct module API path for explicit per-speaker
conditioning.
Each sample should be shaped like `(1, 1, T)` at `24 kHz`.

## Current Generation Config

`VibeVoiceGenerationConfig` currently supports:

- `max_new_tokens`
- `cfg_scale`
- `diffusion_steps`
- `diffusion_steps_fast`
- `diffusion_warmup_frames`
- `do_sample`
- `temperature`
- `top_p`
- `seed`
- `safety_max_new_tokens`

## Script CLI (Advanced)

For `--cfg-scale`, `--diffusion-steps`, `--temperature`, `--seed`, and
multi-speaker `voice_samples`, use the script directly:

- `scripts/generate_vibevoice.py`

Current flags:

- `--text`
- `--model-dir`
- `--tokenizer-dir`
- `--reference-audio`
- `--output`
- `--cfg-scale`
- `--diffusion-steps`
- `--diffusion-steps-fast`
- `--diffusion-warmup-frames`
- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--seed`
- `--greedy`

Important current semantics:

- the core config defaults to deterministic generation
- the current CLI defaults to sampling unless `--greedy` is set
- `--greedy` bypasses sampling controls
- `--seed` matters only when sampling is enabled
- `--diffusion-steps` is the full diffusion budget
- when `--diffusion-steps-fast` is set, the runtime uses the full budget for the
  first `--diffusion-warmup-frames` speech frames, then switches to the reduced
  step count for later frames

Recommended diffusion usage:

- default quality path: omit `--diffusion-steps-fast`
- adaptive speed path: keep `--diffusion-steps` at the quality target, then set
  `--diffusion-steps-fast` lower for later frames
- example: `--diffusion-steps 20 --diffusion-steps-fast 8 --diffusion-warmup-frames 10`

## Recommended Usage

Single-speaker default:

- plain `text`
- no references if you just want the model's default voice behavior

Single-speaker cloning:

- pass `reference_audio`

Multi-speaker:

- prefer `voice_samples=[...]`
- one voice sample per speaker
- speaker-labeled text like `Speaker 1: ...` / `Speaker 2: ...`
- keep one speaker turn per line for the prompt
- use `Speaker N:` labels directly; do not wrap an already-labeled dialogue in an
  extra `Speaker 1: ...` prefix
- `[N]: ...` style tags can be normalized, but `Speaker N:` is the clearest
  documented format

Example multi-speaker prompt:

```text
Speaker 1: Welcome back to the local speech lab.
Speaker 2: Today we are checking the multi-speaker runtime path.
Speaker 3: Keep the turns short and easy to distinguish.
Speaker 4: End the clip before the dialogue starts to drift.
```

## Important Limitation

The current MLX VibeVoice path does **not** yet have a polished multi-speaker
wrapper.

That means:

- multi-speaker with explicit `voice_samples` works better than non-reference use
- non-reference multi-speaker behavior is still weaker and easier to misread
- no-reference multi-speaker prompts should still use explicit newline-separated
  `Speaker N:` turns
- the current CLI only supports one `--reference-audio`, not full multi-speaker
  `voice_samples`

So the best current default for multi-turn / multi-speaker VV is:

- use reference-driven speaker setup
- or add a wrapper that creates synthetic per-speaker defaults

## Current Practical Reality

What we have verified in the local MLX runtime:

- explicit `voice_samples` give stronger speaker anchoring than text-only
  speaker labels
- text-only multi-speaker prompting can work, but it is less stable
- the current CLI is good for single-reference generation, not full
  multi-speaker conditioning

The main follow-on improvements are still:

- a proper multi-speaker wrapper
- optional synthetic default speaker references
- clearer CLI support for multi-speaker `voice_samples`
