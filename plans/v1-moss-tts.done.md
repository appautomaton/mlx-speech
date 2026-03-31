# v1: MOSS-TTS — Local + Delay

**Status: DONE** (2026-03-31)

## Scope

Consolidate the MOSS family into a real MLX-native runtime:

- `MossTTSLocal` for shorter speech, cloning, and continuation
- `MossTTSDelay` / `MOSS-TTSD` for dialogue and multi-speaker generation

This plan closed the gap between the original local-only bring-up and a usable
shipping inference surface for both MOSS model families.

## What Was Delivered

### MossTTSLocal

- quantized-first local runtime with shared Cat codec support
- stable user-facing CLI modes:
  - `generation`
  - `clone`
  - `continuation`
  - `continue_clone`
- locked `clone-v1` preset and fixed clone-eval materialization flow
- continuation and `continue_clone` behavior tightened around prompt packing
- local docs, examples, and published MLX artifacts

### MOSS-TTSD

- full MLX-native `MossTTSDelay` model, processor, and checkpoint loading
- quantized cached runtime at `models/openmoss/moss_ttsd/mlx-int8/`
- end-to-end waveform generation with shared Cat codec decode
- user-facing CLI modes:
  - `generation`
  - `continuation`
  - `voice_clone`
  - `voice_clone_and_continuation`
- JSONL batch path
- real voice-clone and multi-speaker inference on local references
- benchmark helper surface in `scripts/benchmark_moss_ttsd.py`
- docs and published MLX artifacts for the mainline runtime

## Runtime State At Closeout

Mainline MOSS usage is now considered complete and stable enough to close v1:

- `MossTTSLocal` is operational for the shipping inference modes
- TTSD mainline generation works on the local quantized cached path
- the recommended TTSD workflow is `voice_clone_and_continuation`
- Local and TTSD both load from converted MLX weights with the shared codec
- the repo docs and CLIs reflect the runtime people should actually use

## Validation

Focused validation that is in place at closeout includes:

- Local and TTSD config / processor / checkpoint / model tests
- deterministic generation tests
- real-model cached vs uncached greedy TTSD parity on representative prompts
- CLI helper tests
- benchmark helper tests
- quantized checkpoint tests
- source-faithful sampling regressions such as TTSD top-k behavior

## Follow-on Notes

If new issues or deeper investigations come up, they should land in a newer
plan rather than keeping v1 artificially open.

Examples of follow-on work that no longer block this plan:

- deeper sampled-path audits
- broader continuation robustness work outside the recommended workflow
- wider benchmark studies or quantization retuning
- harder multi-speaker drift cases outside normal prompt patterns

## Out of Scope

- VibeVoice (tracked separately in v2)
- Cohere ASR (tracked separately in v3)
- streaming inference
- MOSS-VoiceGenerator, MOSS-SoundEffect, MOSS-TTS-Realtime
- training or finetuning

## Reference Files

| Component | Source |
|-----------|--------|
| Local model | `src/mlx_speech/models/moss_local/` |
| Delay model | `src/mlx_speech/models/moss_delay/` |
| Shared layer | `src/mlx_speech/models/moss_common/` |
| Local generation | `src/mlx_speech/generation/moss_local.py` |
| Delay generation | `src/mlx_speech/generation/moss_delay.py` |
| Upstream MOSS-TTS | `.references/MOSS-TTS/` |
| Upstream MOSS-TTSD | `.references/MOSS-TTSD/` |
