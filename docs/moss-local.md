# MossTTSLocal

## What It Is

`MossTTSLocal` is the smaller OpenMOSS speech model in this repo.

It is the best fit for:

- faster local speech generation
- straightforward cloning
- simpler continuation experiments
- lower-cost runtime compared with TTSD

Current local default runtime:

- `models/openmoss/moss_tts_local/mlx-int8`

Shared codec default:

- `models/openmoss/moss_audio_tokenizer/mlx-int8`

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("moss-local")
result = model.generate("Hello from OpenMOSS TTS Local!")
# result.waveform, result.sample_rate
```

```bash
mlx-speech tts --model moss-local --text "Hello!" -o output.wav
```

Local paths (skips HF download):

```bash
mlx-speech tts \
  --model models/openmoss/moss_tts_local/mlx-int8 \
  --codec models/openmoss/moss_audio_tokenizer/mlx-int8 \
  --text "Hello!" \
  -o output.wav
```

## Script CLI (Advanced)

For advanced sampling controls and multi-mode inference, use the script directly:

- `scripts/generate_moss_local.py`

Supported modes:

- `generation`
- `clone`
- `continuation`
- `continue_clone`

## Mode Semantics

`generation`

- text-only synthesis

`clone`

- uses `--reference-audio` as a voice-conditioning reference

`continuation`

- uses `--reference-audio` as prompt audio context

`continue_clone`

- combines continuation-style prompt audio context with clone-style voice
  conditioning

Current CLI shape:

- single `--reference-audio` input only
- in `continue_clone`, that same reference path is used for both prompt context
  and voice conditioning

## Sampling Controls (Script only)

The unified `mlx-speech tts` CLI exposes `--max-new-tokens` only. The full
sampling surface below is available via `scripts/generate_moss_local.py`.

Local exposes separate text and audio sampling controls:

- `--text-temperature`
- `--text-top-k`
- `--text-top-p`
- `--text-repetition-penalty`
- `--audio-temperature`
- `--audio-top-k`
- `--audio-top-p`
- `--audio-repetition-penalty`
- `--greedy`

Current default runtime:

- KV cache on for eligible single-sample paths
- `--no-kv-cache` is available as a debug fallback

## Duration Controls (Script only)

Local also exposes duration-style prompt controls:

- `--expected-tokens`
- `--auto-estimate-expected-tokens`

Important current behavior:

- duration control applies to `generation` and `clone`
- continuation-style modes ignore expected tokens
- `--no-max-new-tokens` still keeps the internal safety cap

## Important Continuation Note

For continuation-style use, the continuation text should match the prompt audio
context. A mismatched prompt transcript can produce weak or silent continuation
behavior.

Also:

- continuation outputs usually contain the new continuation segment
- they do not necessarily replay the prompt audio transcript verbatim

## Output Polishing Flags (Script only)

Current convenience flags:

- `--trim-leading-silence`
- `--normalize-peak`

These are post-processing conveniences, not core model behavior.
