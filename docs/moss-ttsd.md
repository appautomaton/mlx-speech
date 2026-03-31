# MOSS-TTSD

## What It Is

`MOSS-TTSD` is the large OpenMOSS dialogue model built on `MossTTSDelay`.

It is the best fit in this repo for:

- multi-speaker dialogue
- turn-taking audio
- voice-cloned dialogue
- longer conversational synthesis

Current local default runtime:

- `models/openmoss/moss_ttsd/mlx-int8`

Shared codec default:

- `models/openmoss/moss_audio_tokenizer/mlx-int8`

## Main CLI

Script:

- `scripts/generate_moss_ttsd.py`

I/O shapes:

- single-sample mode writes one WAV
- batch mode reads JSONL sequentially and writes `output.jsonl` plus WAV files

Supported modes:

- `generation`
- `continuation`
- `voice_clone`
- `voice_clone_and_continuation`

Current default behavior:

- quantized runtime by default
- KV cache on by default
- `max_new_tokens` defaults to `2048`

## Mode Requirements

`generation`

- text only

`voice_clone`

- `text`
- at least one `prompt_audio_speakerN`

`continuation`

- `text`
- at least one paired `prompt_audio_speakerN` and `prompt_text_speakerN`

`voice_clone_and_continuation`

- same paired speaker prompt requirements as continuation-style use
- uses those paired prompts for both continuation context and speaker anchoring

## Sampling Controls

Current user-facing audio controls:

- `--temperature`
- `--top-p`
- `--top-k`
- `--repetition-penalty`
- `--greedy`

Practical note:

- `--greedy` disables sampling
- otherwise TTSD uses the audio sampling controls above
- `--no-kv-cache` is available as a debug fallback

Useful CLI helpers:

- `--text-normalize`
- `--sample-rate-normalize`

## Recommended Usage

For real dialogue work:

- use explicit speaker tags like `[S1]`, `[S2]`
- prefer clone-style conditioning for speaker stability
- give enough row budget for the script

Practical budgeting:

- short prompt: around `128` to `256`
- medium dialogue: around `384` to `640`
- long dialogue: larger, intentional budget

The script reports actual emitted rows separately from the cap.

## Current Strengths

- strongest multi-speaker path in the repo
- working quantized runtime
- working KV cache
- working clone and continuation surfaces

## Current Weak Spots

- long multi-speaker drift still needs hardening
- continuation quality still benefits from careful prompt design
- some quality issues are conditioning-related, not architecture-related
