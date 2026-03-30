# v1: MossTTSLocal Follow-On Work

## Scope

This file no longer plans the first implementation of voice cloning.
Those core inference capabilities are already in the main MLX runtime:

- direct text-to-speech
- clone
- continuation
- continuation + clone
- processor-side reference audio encode/decode helpers
- CLI support for the four main inference modes

v1 now tracks the follow-on work needed to make those capabilities stronger:
quality, parity, performance, ergonomics, and release readiness.

Stay on `MossTTSLocal` inference only. Do not add training, finetuning, or
new model families here.

## Current Reality

The current runtime already provides:

- pure-MLX speech generation on `MossTTSLocal`
- pure-MLX Cat codec encode + decode
- local-path-first loading from:
  - `models/openmoss/moss_tts_local/mlx-int8/`
  - `models/openmoss/moss_audio_tokenizer/mlx-int8/`
- default `W8Abf16` mixed-precision runtime:
  - quantized weights on supported modules
  - `bfloat16` hidden-state path
  - `float32` preserved for numerically sensitive reductions
- default global + local KV cache for single-item sampled inference:
  - `~2.32x` speedup over uncached path
  - `~1.64x` faster than real time on quantized hardware
  - uncached path retained as `--no-kv-cache` fallback
- processor parity for the four main conversation shapes
- end-to-end inference for generation / clone / continuation / continue_clone

This means v1 is not "make cloning work." It is "make cloning solid."

## Main Gaps

### 1. Longer Parity Windows

Current parity work is enough to trust bring-up, but not enough to declare
full sequence equivalence.

Need:
- extend greedy rollout parity beyond the current short validated window
- compare original vs quantized runtime for longer direct-generation runs
- add reference-audio parity checks for clone and continuation modes
- identify where drift begins when it does appear

### 2. Runtime Performance

The main structural performance costs are now addressed:
- global KV cache: implemented, default on
- local RVQ depth KV cache: implemented, default on
- preallocated decode buffers: no `mx.concatenate` in the hot loop
- benchmarked at `~2.32x` speedup, `~1.64x` faster than real time (quantized)

Remaining performance work:
- Python-loop overhead around the hierarchical generation loop
- benchmarking consistency across short / medium / long prompts
- memory footprint measurement on representative workloads

### 3. Quality Controls

Core audio quality is now good enough to be useful, but there are still user-
visible behaviors that need stronger control:

- stop behavior on longer prompts
- duration conditioning stability
- pause length between sentences
- tail cleanup around EOS
- clone/continuation quality consistency on real reference audio

This stage should produce:
- a small set of known-good generation defaults
- quality regression prompts
- clearer expected behavior for short vs long prompts

### 4. API and CLI Hardening

Current interfaces work, but they still reflect bring-up history.

Need:
- stabilize the conversation-oriented inference API
- keep the text-only helper as a thin convenience layer
- document mixed-batch behavior clearly
- add example scripts for:
  - direct TTS
  - clone
  - continuation
  - continuation + clone
- keep output file handling and artifact naming predictable

### 5. Artifact and Release Preparation

The runtime now depends on local quantized artifacts. Release work still needs:

- final verification of the regenerated codec `mlx-int8/` artifact
- publication plan for converted weights to Hugging Face
- README examples that use the quantized runtime by default
- explicit versioned benchmark notes for speed / quality / memory

## Proposed Stages

### Stage 1 — Parity Hardening

- extend direct-generation greedy parity to longer windows
- add original-vs-quantized runtime comparisons on the same prompts
- add clone / continuation parity probes at processor and waveform levels
- record first divergence points when they exist

### Stage 2 — Performance Pass

Global + local KV cache is implemented and default. Remaining:
- benchmark `W8Abf16` runtime across short / medium / long prompts
- track wall-clock latency, audio seconds per second, memory footprint
- identify any Python-loop overhead in the hierarchical generation loop

### Stage 3 — Quality Pass

- tighten stop behavior and prompt-length heuristics
- reduce pathological long pauses
- validate clone / continuation on a small fixed listening set
- define a small "known good" preset for end-user CLI generation

### Stage 4 — Interface Cleanup

- lock the public inference surface
- add examples and docs for each mode
- remove bring-up-oriented ambiguity from CLI help and examples

### Stage 5 — Release Readiness

- prepare Hugging Face publication workflow for MLX artifacts
- finalize local model layout documentation
- document runtime precision policy (`W8Abf16` mixed precision)

## Done Criteria

v1 is done when:

- direct generation, clone, continuation, and continuation + clone remain
  pure-MLX and stable on the quantized runtime
- longer parity checks are recorded and understood
- performance work materially improves current decode speed
- the default quantized runtime is documented as the primary path
- examples and docs reflect the real public inference surface

## Out of Scope

- training or finetuning
- non-Moss model families
- `MOSS-TTSD`
- server deployment work
- streaming inference if it requires large architectural changes
