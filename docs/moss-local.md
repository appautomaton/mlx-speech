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

## Main CLI

Script:

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

## Sampling Controls

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

## Duration Controls

Local also exposes duration-style prompt controls:

- `--expected-tokens`
- `--auto-estimate-expected-tokens`

Important current behavior:

- duration control applies to `generation` and `clone`
- continuation-style modes ignore expected tokens

## Important Continuation Note

For continuation-style use, the continuation text should match the prompt audio
context. A mismatched prompt transcript can produce weak or silent continuation
behavior.

Also:

- continuation outputs usually contain the new continuation segment
- they do not necessarily replay the prompt audio transcript verbatim

## Output Polishing Flags

Current convenience flags:

- `--trim-leading-silence`
- `--normalize-peak`

These are post-processing conveniences, not core model behavior.
