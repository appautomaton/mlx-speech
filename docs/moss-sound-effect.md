# MOSS-SoundEffect

## What It Is

`MOSS-SoundEffect` is the OpenMOSS sound-effect checkpoint built on the shared
`MossTTSDelay` architecture.

It is for:

- ambience
- environmental sound
- action sounds
- sound-design style clips

It is **not** a speech model.

Current local default runtime:

- `models/openmoss/moss_sound_effect/mlx-4bit`

Shared codec default:

- `models/openmoss/moss_audio_tokenizer/mlx-int8`

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("moss-sound-effect")
result = model.generate("rolling thunder with steady rainfall", duration_seconds=6)
# result.waveform, result.sample_rate
```

```bash
mlx-speech tts --model moss-sound-effect \
  --text "rolling thunder with steady rainfall" \
  --duration-seconds 6 \
  -o output.wav
```

Local paths (skips HF download):

```bash
mlx-speech tts \
  --model models/openmoss/moss_sound_effect/mlx-4bit \
  --codec models/openmoss/moss_audio_tokenizer/mlx-int8 \
  --text "rolling thunder with steady rainfall" \
  --duration-seconds 6 \
  -o output.wav
```

Note: the unified CLI uses `--text` for the ambient sound description.

## Script CLI (Advanced)

For sampling controls and `--expected-tokens` tuning, use the script directly:

- `scripts/generate_moss_sound_effect.py`

Current runtime defaults:

- device `auto`, which resolves to GPU
- KV cache on unless `--no-kv-cache` is set

Core required input:

- `--ambient-sound`

Important current prompt rule:

- ambient sound description only
- no reference audio

## Duration Controls

Current controls:

- `--duration-seconds`
- `--expected-tokens`
- `--max-new-tokens`

Behavior:

- `duration_seconds` is converted to expected tokens
- the heuristic is `12.5 tokens / second`
- `max_new_tokens` is still the hard cap

So duration is guided, not frame-perfect.

## Sampling Controls (Script only)

The unified `mlx-speech tts` CLI exposes `--duration-seconds` and
`--max-new-tokens`. Full sampling controls below are available via
`scripts/generate_moss_sound_effect.py`.

Current user-facing controls:

- `--temperature`
- `--top-p`
- `--top-k`
- `--repetition-penalty`
- `--greedy`

Current defaults follow the upstream SoundEffect recipe:

- temperature `1.5`
- top-p `0.6`
- top-k `50`
- repetition penalty `1.2`

## Recommended Prompting

Good prompts are:

- concrete
- scene-oriented
- sound-focused

Examples:

- `rolling thunder with steady rainfall on a metal roof`
- `clear footsteps echoing on concrete at a steady rhythm`
- `early morning park ambience with light birds chirping and a gentle breeze`

Less reliable use:

- true music generation

The model can produce music-like texture, but it is better treated as a sound
effect / ambience generator than a dedicated music model.
