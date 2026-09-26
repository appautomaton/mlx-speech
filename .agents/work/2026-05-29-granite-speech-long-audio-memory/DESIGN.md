# Granite Speech Long-Audio Memory Design

## Problem

The current Granite LM attention path hand-materializes fp32 `[B, heads, L, L]`
attention scores and weights during prefill. For 270-second chunks, `L` is
about 2,718, which makes each layer allocate large temporary matrices. That is
consistent with the observed 100+ GB system memory pressure. The KV cache is
not the largest allocation, but it is still currently over-allocated as fp32.

## Runtime Changes

### Attention

- Replace the manual `q @ k.T`, causal mask, softmax, and `weights @ v` path
  with `mx.fast.scaled_dot_product_attention`.
- Pass KV heads without pre-tiling. MLX fast attention supports GQA/MQA when
  `q` has more heads than `k`/`v`.
- Use `mask="causal"` for prefill/full forward. MLX uses lower-right causal
  alignment, which also works for one-token cached decode.
- Preserve Granite's configured `attention_multiplier` as the attention scale.

### KV Cache

- Allocate KV cache with the active runtime dtype where possible, not hard-coded
  fp32.
- Preserve exact cache length semantics:
  - prefill length equals prompt length
  - each decode step increments by one token
  - overflow remains a clear `ValueError`
- Keep KV cache enabled; disabling it is diagnostic-only and out of scope.

### Context Preflight

- For audio paths or arrays, compute `audio_tokens` from sample count before
  STFT/encoder/projector execution.
- Build prompt IDs from that count and reject over-context requests before
  `get_audio_features(...)` can allocate encoder/projector intermediates.

## Memory Telemetry

Use coarse MLX snapshots only:

- `mx.get_active_memory()`
- `mx.get_cache_memory()`
- `mx.get_peak_memory()`
- optional `mx.reset_peak_memory()` at benchmark boundaries
- optional `mx.clear_cache()` at cleanup boundaries

Telemetry belongs in diagnostic/benchmark summaries. It must not poll per token
or run a background sampler.

## Long-Audio Benchmark

Use `/tmp` for media, scripts, chunks, transcripts, and summaries. The default
benchmark source is LibriVox plus Project Gutenberg for Jean M. Thompson's
public-domain *The Three Bears of Porcupine Ridge*, chapter 6, "Tracked by a
Catamount".

The model cannot process the whole 17:49 file in one prompt because the prompt
exceeds the 4,096-token context. The benchmark must therefore use context-safe
chunking, then compare the concatenated transcript with the known chapter text
using normalized word-level metrics.

## Verification Strategy

- Unit tests prove attention/cache semantics and no explicit attention matrix
  implementation remains.
- Unit tests prove preflight failure happens before model audio feature
  projection.
- Unit tests prove telemetry and benchmark summary shape without network or
  real model weights.
- Existing Granite unit/checkpoint/runtime gates must continue to pass.
- A real `/tmp` long-audio benchmark run provides final behavioral evidence.
