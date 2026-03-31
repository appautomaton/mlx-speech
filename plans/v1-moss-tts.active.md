# v1: MOSS-TTS — Local + Delay

## Scope

Consolidation of all MOSS-TTS model work:
- MossTTSLocal hardening (quality, parity, performance, release)
- MossTTSDelay (MOSS-TTSD 8B) implementation for multi-speaker dialogue

v0 delivered working MossTTSLocal end-to-end inference. v1 now tracks both:
- MossTTSLocal hardening
- MossTTSDelay inference bring-up and hardening

This active plan is now the source of truth for TTSD work. Treat the old
standalone TTSD draft as superseded planning context, not the active execution
plan. The older split follow-on files for clone-only and TTSD-only planning are
historical notes now; do not treat them as separate active plans.

Do not port MOSS-VoiceGenerator, MOSS-SoundEffect, or MOSS-TTS-Realtime.

## Part A: MossTTSLocal Hardening

### What's Already Done

Stage 1 (Quality Pass) is complete:
- `continuation` and `continue_clone` reject `tokens` conditioning
- fixed clone eval set: 4 macOS voices × 3 prompts (English)
- `clone-v1` preset locked: `temp=1.0, top_p=0.95, top_k=50, rep_pen=1.1`
- sweep script and materialization workflow operational
- quantized local runtime is the default path for both `MossTTSLocal` and the
  shared Cat codec
- user-facing Local CLI supports:
  - `generation`
  - `clone`
  - `continuation`
  - `continue_clone`

### Remaining Work

**A1 — Parity Hardening:**
- extend greedy rollout parity beyond the current 17-row window
- original-vs-quantized runtime comparisons on the same prompts
- clone / continuation parity probes at processor and waveform levels
- record first divergence points when they exist

**A2 — Performance Benchmarking:**
- benchmark `W8Abf16` across short / medium / long prompts
- track wall-clock latency, audio seconds per second, memory footprint
- identify any Python-loop overhead in the hierarchical generation loop

**A3 — Interface Cleanup:**
- lock the public inference surface
- add examples and docs for each mode
- remove bring-up-oriented ambiguity from CLI help

**A4 — Release Readiness:**
- `mlx-int8` artifacts published to HuggingFace
- finalize local model layout documentation
- document runtime precision policy

## Part B: MossTTSDelay (MOSS-TTSD 8B)

### Architecture

Single Qwen3 8B backbone predicts all RVQ codebook heads in parallel via
delay scheduling. No local transformer.

| Aspect | MossTTSDelay | MossTTSLocal |
|--------|-------------|-------------|
| RVQ prediction | Parallel (all heads per forward) | Sequential (depth-first) |
| Architecture | Single Qwen3 8B | Global Qwen3 1.7B + 4-layer local |
| Forward passes per timestep | 1 | 1 + 32 |
| n_vq | 16 | 32 |

### What's Already Done

- `MossTTSDelayConfig` implemented
- `MossTTSDelayModel`: `language_model` + `emb_ext` + `lm_heads`
- `MossTTSDelayProcessor`: delay/de-delay round-trip, placeholder expansion,
  reference audio packing, de-delay before codec decode
- `moss_common/` shared layer (Qwen3 backbone, base processor/tokenizer)
- explicit-path loading still supports original-style TTSD checkpoints for
  parity/debug work when such a path is provided manually
- MLX-native TTSD `mlx-int8` conversion landed at
  `models/openmoss/moss_ttsd/mlx-int8/`
- TTSD runtime is now quantized-first by default with explicit `--model-dir`
  / `--codec-dir` overrides instead of hidden local-original fallback
- uncached generation loop with delay-aware state machine
- cached generation loop now implemented truthfully for the single Qwen3
  backbone via `prefill(...)` + `decode_step(...)`
- parity audit framework against upstream Torch
- tests now cover:
  - config / processor / checkpoint / model
  - deterministic fake-model cached vs uncached generation parity
  - real-model greedy cached vs uncached parity on short and medium prompts
  - quantized checkpoint round-trip
  - CLI and benchmark helper behavior
- first smoke output: non-empty WAV, correct sequence structure
- greedy source parity now holds on audited short windows for both
  punctuation-free and punctuation-bearing generation prompts
- one real sampled-path source mismatch has been found and fixed:
  TTSD `top-k` filtering now keeps the exact top-K indices like upstream
- additional source-faithful sampling/runtime fixes landed:
  - real `-inf` masking
  - zero-temperature forces greedy
  - no text-head repetition penalty in TTSD sampled mode
  - cached tail-head sampling keeps the uncached flattened semantics
- short punctuation-bearing generation prompts now terminate cleanly on the
  MLX GPU path after the top-k fix
- current cached TTSD hot loop no longer grows sequence buffers with per-step
  `mx.concatenate(...)`; it now uses:
  - preallocated sequence buffers
  - device-side delay state tensors
  - device-side sampling masks
  - fewer Python-side sync points
- TTSD dialogue helpers now mirror the upstream continuation/clone workflow:
  - speaker-tag normalization
  - per-speaker prompt audio/text collection
  - `prepare_ttsd_sample(...)`
  - sequential JSONL batch input handling
- TTSD user-facing CLI now supports:
  - `generation`
  - `continuation`
  - `voice_clone`
  - `voice_clone_and_continuation`
- TTSD voice-clone path is exercised end to end with real local reference audio
- multi-speaker TTSD samples now run on the quantized cached path in practical
  wall-clock time on the local machine
- dedicated TTSD benchmark surface now exists at
  `scripts/benchmark_moss_ttsd.py` and reports:
  - load time
  - prepare time
  - generation-only time
  - decode time
  - end-to-end time

### Current State

The main TTSD runtime path is real and end-to-end:

```text
text / prompt audio
  -> processor
  -> MossTTSDelay
  -> delay-pattern generation loop
  -> de-delay
  -> Cat codec decode
  -> waveform
```

What is now true:
- local quantized TTSD checkpoints strict-load and are now the only default
  runtime expected on disk
- local quantized Cat codec checkpoints strict-load and are the default shared
  codec path for both Local and TTSD
- TTSD generation mode on the MLX GPU is practical for short and medium
  single-speaker prompts
- cached and uncached greedy rows match on real short and medium punctuation
  prompts
- short generation without punctuation works
- short and medium punctuation-bearing prompts now work on the MLX GPU path
- a five-sentence cached quantized generation sample now completes cleanly on
  the current local machine in a practical runtime window
- TTSD multi-speaker voice-clone samples now run end to end on the quantized
  cached path
- processor packing and greedy generation are aligned enough that TTSD
  debugging is no longer dominated by processor/state-machine uncertainty
- the active speed work has moved from basic bring-up to runtime efficiency
  tuning on the cached quantized path

What is not yet trusted enough:
- sampled-path parity beyond the currently audited short slices
- continuation quality / stop behavior on representative prompts
- broader TTSD prompt robustness, especially harder multi-speaker drift cases
- current TTSD quantization recipe as the final speed-optimized recipe
- full benchmark-matrix conclusions across short / medium / long prompts

### Remaining Work

**B1 — Continuation + Sampled Hardening:**
- continue hardening stop behavior on representative prompts
- extend beyond the current working generation prompts to continuation-style
  prompts
- keep auditing TTSD sampled generation strictly against upstream source logic
- continue checking for any missing or simplified inference-time logic in:
  - masking
  - repetition penalty
  - top-k
  - top-p
  - state updates
- prefer source-level logic audit over repeated blind numeric rechecks

**B2 — Performance Benchmarking + Hot-Loop Tuning:**
- use `scripts/benchmark_moss_ttsd.py` as the authoritative timing surface
- benchmark the cached quantized runtime on short / medium / long prompts
- compare:
  - quantized cached vs quantized uncached
  - quantized cached vs any explicit reference checkpoint path when one is
    intentionally provided for comparison
- keep optimizing the cached hot loop only if generation-only timing still
  leaves obvious wins on the table

**B3 — TTSD Quantization Recipe Tuning:**
- keep any explicit original/reference TTSD path unchanged when one is used for
  correctness comparison
- tune the current TTSD quantization recipe only if the benchmark surface shows
  it is still slower than the best available reference/runtime comparison on
  representative prompts
- likely recipe order:
  1. current baseline
  2. backbone-only quantization
  3. backbone + large heads, leaving smaller TTSD-specific modules unquantized

**B4 — Multi-Speaker Quality Hardening:**
- keep testing multi-speaker clone and continuation prompts for speaker drift
- validate the recommended `voice_clone_and_continuation` path on representative
  local references
- tighten guidance on prompt length, speaker turn structure, and reference
  quality now that the basic multi-speaker path is implemented

**B5 — Runtime + Publishing Cleanup:**
- finalize docs around quantized-only local defaults
- publish the converted MLX artifacts to HuggingFace
- keep explicit-path loading available for any future custom checkpoints

Completed milestones that no longer belong in Remaining Work:
- TTSD `mlx-int8` conversion
- TTSD cached generation bring-up
- TTSD multi-speaker pipeline scaffolding
- TTSD JSONL batch mode
- TTSD clone / continuation / voice_clone_and_continuation CLI surface

Archived bring-up note for context only:
- audit TTSD sampled generation strictly against upstream source logic
- the first TTSD execution plan treated `mlx-int8` and KV cache as future work;
  they are now done and should not be tracked as open bring-up items

## Validation

Current focused validation for TTSD remains green, including:
- config / processor / checkpoint / model tests
- deterministic fake-model generation tests
- real-model cached vs uncached greedy parity on short and medium prompts
- CLI helper tests
- benchmark helper tests
- quantized checkpoint tests
- source-faithful top-k regression coverage

## Done Criteria

v1 is done when:
- MossTTSLocal clone produces consistent output on the fixed eval set
- parity checks extended and recorded
- `mlx-int8` artifacts published to HuggingFace for both Local and TTSD
- MossTTSDelay generates multi-speaker dialogue from script input
- TTSD output is 24 kHz waveform with distinct speaker voices
- both models load from converted int8 MLX weights
- CLI supports all modes for both models

## Out of Scope

- VibeVoice (that's v2)
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
