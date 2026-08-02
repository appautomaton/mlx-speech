# dots.tts Bounded Caching and Waveform Streaming

**Bet:** Request-local bounded caches can remove dots.tts history recomputation and expose useful waveform streaming on Apple Silicon without changing checkpoints, cloning behavior, or the unified non-streaming result.

## Bounded Goal

Replace the current concatenate-and-recompute dots.tts inference path with bounded MLX-native Qwen, semantic, DiT, and vocoder state, then expose optional waveform streaming while preserving `generate()` behavior and proving at least a twofold long-generation speedup for both MeanFlow and SOAR under a 30 GiB peak-memory limit.

## Classification

- Work scale: capability
- Work shape: mixed — inference optimization, refactor, public API addition, and regression coverage
- Selected lenses: product, engineering, runtime

## Target Stakeholder

Apple Silicon developers using dots.tts through `mlx-speech` who need lower long-generation latency, bounded memory behavior, and incremental waveform delivery without adding a Torch or `mlx-lm` runtime.

## Broader Intent

The change keeps dots.tts aligned with the library's pure-MLX mission while establishing a reusable optional streaming capability for TTS families. It must not force non-streaming families to imitate streaming or expose upstream-specific runtime abstractions as the common API.

## Approved Approach and Evidence

The approved approach combines bounded inference caches with public dots.tts waveform streaming:

- The pinned official checkout at `.references/dots.tts` implements separate bounded Qwen, semantic, per-NFE DiT, recurrent vocoder, and decoder-window state. It establishes the intended algorithm but does not prove MLX performance or numerical parity.
- The current MLX runtime already performs correct incremental Qwen and semantic inference, but grows their K/V state by concatenation. Its DiT recomputes the full acoustic history for every solver evaluation, and its vocoder `decode_chunk` retains and decodes all prior latents.
- MLX 0.31.1 was locally verified to support preallocated slice writes on Metal. The implementation may use this mechanism without adding `mlx-lm` as a dependency.
- The user approved an optional `StreamingTTSModel` capability, `TTSOutput` chunks, the official hybrid vocoder cadence, a configurable `stream_chunk_patches=4` default, a 500-patch default budget with a 512 hard maximum, and a relative twofold performance gate.

The uncached DiT path remains an internal oracle and benchmark baseline. Cache selection is not a public control and the optimized path must not silently fall back to full-history recomputation.

## Required Outcome

### REQ-001 — Bounded Qwen and semantic state

- Qwen and the dots.tts semantic encoder append K/V into capacity-managed MLX arrays with explicit valid lengths instead of concatenating the complete cache for every new token or patch.
- The shared Qwen change preserves VibeVoice behavior, cache dtypes, causal positions, and incremental/full-forward parity.
- Request state has a deterministic capacity bound and raises a clear error on overflow.

### REQ-002 — Per-NFE DiT cache

- MeanFlow and SOAR keep distinct K/V state for every solver evaluation and Transformer layer; SOAR additionally separates its conditional and unconditional branches.
- The first patch runs without history. Later patches read finalized persistent history from cache, recompute only the previous finalized unit plus the current hidden/noisy unit, and commit only the previous unit after each solver evaluation.
- Prompt-continuation history is prefetched separately for each solver evaluation. Speaker conditioning, positions, causal masks, ODE schedules, and adaptive modulation remain equivalent to the full-history reference path.
- Cache capacity uses the official 64, 128, 256, and 512 patch buckets. The current four-latent patch plus one hidden token remains one five-token DiT unit.
- The current noisy latent tail is never persisted, and lazy MLX graphs do not retain prior patch work after a patch is materialized.

### REQ-003 — Incremental vocoder state

- AudioVAE decoding carries the decoder-side SLSTM hidden and cell state between chunks.
- BigVGAN re-decodes only a bounded window derived from its finite left context, decoder lookahead, and maximum input chunk.
- The first two payload patches are decoded separately; later payload patches are merged according to `stream_chunk_patches`, defaulting to four. Remaining patches and lookahead are flushed at normal completion.
- Concatenated default stream output remains numerically equivalent to the trusted batch-decoder behavior and finite, non-silent, mono 48 kHz audio.

### REQ-004 — Optional public TTS streaming

- Export `StreamingTTSModel` as an optional extension of `TTSModel` with `generate_stream(...) -> Iterator[TTSOutput]`.
- Only dots.tts implements it in this change. Other adapters and the return type of `tts.load(...)` remain compatible.
- dots.tts `generate_stream()` accepts the same voice-cloning and generation controls as `generate()`, plus positive `stream_chunk_patches`.
- Every yielded `TTSOutput` contains a non-empty one-dimensional float32 waveform and the 48,000 Hz sample rate. Iterator exhaustion marks completion; no final-marker type is added.
- `generate()` consumes the same internal stream at the default cadence and concatenates chunks, so the public non-streaming path does not maintain separate generation logic.
- Closing an iterator early stops work and releases request-local state without computing or emitting a flush chunk.

### REQ-005 — Safe bounds and compatibility

- Preserve `max_audio_patches=500` as the public default and reject values above 512. Examples continue to recommend 128 as a conservative starting budget.
- Runtime K/V caches use the model's activation dtype. The existing selective int8 weight policy remains unchanged; cache quantization is not introduced.
- No checkpoint schema, conversion output, model tensor, model alias, or Hugging Face artifact layout changes.
- Runtime remains pure MLX and adds no Torch, torchaudio, Transformers, `mlx-lm`, `mlx-audio`, or upstream-package dependency.

### REQ-006 — Reproducible performance and documentation

- Add a cached-versus-uncached dots.tts benchmark that records total generation time, time to first non-empty waveform chunk, output duration, RTF, patch count, and MLX peak memory.
- Update the dots.tts guide and Hugging Face model-card source to describe streaming, cache bounds, latency/throughput control, measured results, and the corrected 500-patch default.
- Preserve the historical quantization report. Remote model-card publication and all weight uploads remain outside this change.

## Acceptance Criteria

1. Qwen and semantic incremental outputs match their full/reference paths within the existing dtype-aware tolerances, cache offsets advance correctly, and no token-by-token full-cache concatenation remains in their decode paths.
2. VibeVoice shared-Qwen unit coverage passes without changing its public behavior or mixed cache dtypes.
3. Tiny deterministic MeanFlow and SOAR tests show cached and full-history solvers agree for first patch, multiple subsequent patches, continuation prefill, speaker conditioning, and CFG; each NFE owns separate state and the noisy tail is absent from persistent K/V.
4. The DiT persistent cache grows by exactly five tokens per finalized unit, respects 64/128/256/512 buckets, rejects requests above 512 patches, and remains request-local.
5. SLSTM chunked output agrees with full-sequence output, the vocoder window remains bounded, partial merge and final lookahead flush are correct, and default streamed chunks concatenate to the trusted waveform result.
6. `StreamingTTSModel` is exported, dots.tts structurally implements it, other adapters remain unchanged, and public `generate()` equals the concatenation of the default `generate_stream()` result.
7. Existing SOAR/MF × base/int8 × continuation/speaker-only quality coverage stays within absolute WER regression `0.01` and speaker-cosine regression `0.02` of the published gate.
8. On the same Apple host, Hank speaker-only input, seed 42, default solver steps, `eos_threshold=1.0`, and exactly 128 payload patches, the median of three cached runs is at least 2.0 times faster than the uncached path for both MeanFlow and SOAR; checkpoint loading is excluded from timing.
9. Each performance case yields a non-empty chunk before full completion and stays below 30 GiB peak MLX memory. A 512-budget one-patch smoke proves the maximum cache allocation also remains below the limit.
10. `pytest tests/unit/`, relevant checkpoint/runtime tiers, local dots.tts integration, and the fixed quality gate pass outside the sandbox on Metal before completion.
11. Documentation states the 500 default, 512 maximum, 128 example recommendation, optional streaming API, hybrid cadence, measured performance, and current long-text limitation without making unmeasured real-time claims.

## Constraints and Risks

- DiT cache entries depend on ODE timestep and adaptive conditioning. Reusing K/V across NFE indices would be incorrect even when token positions match.
- SOAR duplicates cache state for classifier-free guidance. At the 512 bucket, logical BF16 DiT K/V is approximately 3.52 GiB versus 0.70 GiB for MeanFlow; model weights and transient activations remain additional memory.
- MLX lazy evaluation can retain computation graphs unless state updates are materialized at controlled boundaries.
- The Qwen cache is shared with VibeVoice; an internal type change has cross-family regression risk.
- Bounded vocoder decoding must reproduce initial padding, causal left context, recurrent state, and final lookahead. An undersized window can create boundary artifacts without raising an exception.
- The twofold target is a measured release gate, not an assumption. A result below it fails verification and returns to planning rather than weakening the threshold.

## Scope Coverage

### Included

- Bounded Qwen, semantic, per-NFE DiT, recurrent SLSTM, and decoder-window state.
- Public synchronous dots.tts waveform streaming through an optional common protocol.
- Configurable stream merge size, compatibility tests, performance benchmark, memory guard, docs, and model-card source updates.

### Deferred / Not in scope

- Automatic sentence splitting and cross-segment waveform stitching. The request-local state and iterator boundary must leave room for a later outer segment scheduler.
- Async streaming, live audio playback, and a streaming CLI.
- KV-cache quantization, cross-request prompt-cache reuse, persistent cache serialization, or paged/rotating context eviction.
- Remote Hugging Face mutations and checkpoint republishing, because runtime weights and artifact layout do not change.

## Anti-goals

- Do not import, vendor, or execute upstream PyTorch inference code.
- Do not describe buffered full-history decoding as streaming.
- Do not force non-streaming TTS families to emit fake chunks.
- Do not expose upstream cache objects, solver state, or cache on/off controls through the unified public API.
- Do not silently continue with the uncached DiT path after an optimized-path error.
- Do not broaden this change into arbitrary-length text orchestration or a new serving framework.
