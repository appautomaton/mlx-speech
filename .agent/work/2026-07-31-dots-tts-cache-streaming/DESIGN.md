# dots.tts Cache and Streaming Design

## Context

The current runtime has two incremental Transformer paths but stores their history by concatenating full K/V arrays. The acoustic DiT has no cache and reruns every historical token for every ODE evaluation. `AudioVAE.decode_chunk` retains and decodes all latents, so it is buffered equivalence rather than bounded streaming.

The design must preserve the existing checkpoints, selective int8 policy, cloning modes, solver math, and non-streaming API while making request memory predictable.

## Request data flow

```text
text/reference -> Qwen cache -> current hidden --------------------+
                      ^                                           |
                      |                                           v
generated latent -> semantic cache                    cached DiT solver
                                                              |
                                                              v
                                                        latent patch
                                                              |
                                                              v
                                          stateful SLSTM + decoder window
                                                              |
                                                              v
                                                      waveform chunks
```

Every mutable object belongs to one generation iterator. The model stores weights and immutable precomputed helpers only; it never retains request cache state.

## Capacity-managed append cache

Qwen and the semantic encoder use one small internal cache abstraction with:

- preallocated key/value arrays;
- an integer valid offset;
- an explicit maximum capacity;
- slice assignment for append;
- valid-prefix views for attention.

Qwen grows in 256-token blocks when the caller does not provide an exact bound, matching the established MLX cache pattern. dots.tts supplies its known schedule bound. The semantic encoder supplies the audio-patch budget. Growth never exceeds the model/request maximum and overflow raises `ValueError`.

The cache preserves projected key and value dtypes independently. This matters to VibeVoice, whose shared Qwen dtype policy is not uniformly BF16.

## DiT delayed-commit cache

One finalized acoustic unit contains one hidden token and four latent tokens. The DiT cache uses buckets of 64, 128, 256, or 512 units and stores K/V conceptually as:

```text
[nfe, layer, branch_batch, head, capacity_tokens, head_dim]
```

MeanFlow has one branch and four default NFE entries. SOAR has conditional and unconditional branches and ten default NFE entries. Cache entries cannot cross NFE indices because timestep and AdaLN conditioning change their values.

For the first patch, the DiT processes only current hidden plus noisy latent. For each later patch it processes:

```text
previous finalized unit + current hidden + current noisy latent
```

Attention reads an already-finalized persistent prefix from cache. Queries in the previous unit follow causal order; the current unit can attend to the persistent prefix and the full fresh tail. After each NFE, only K/V for the previous unit is written to that NFE's cache slot. The current noisy latent is never committed.

Continuation prompt history is prefetched separately for every NFE. The final prompt unit stays fresh for the first generated step, matching the same delayed-commit rule.

The cached runner assembles fused QKV projections once from loaded weights and precomputes the ODE schedule, rotary table, speaker-conditioned AdaLN values, and masks needed by the request. Stored checkpoint names and tensors do not change. The existing full-history solver remains an internal oracle and benchmark baseline.

## Incremental AudioVAE decode

The decoder-side SLSTM exposes a chunk step that accepts and returns per-layer hidden and cell arrays. Its existing full-sequence call starts from zeros and delegates to the same recurrence.

After SLSTM, BigVGAN receives a bounded latent window:

```text
maximum incoming chunk + decoder lookahead + finite decoder left context
```

Left context is derived from local convolution, causal transposed-convolution, AMPBlock, and alias-free activation parameters. The decoder emits only frames that are stable outside its lookahead. Normal completion flushes the final lookahead; iterator cancellation performs no flush.

The default cadence decodes payload patches 1 and 2 separately, then groups four patches. `stream_chunk_patches` changes only the later group size.

## Public boundary

`StreamingTTSModel` extends `TTSModel` structurally and returns `Iterator[TTSOutput]`. dots.tts is the only implementation in this change. `TTSOutput` remains unchanged and iterator exhaustion is the completion signal.

`DotsTTSAdapter.generate()` drains the default stream and concatenates waveform chunks. Cache policy and internal request state are not public arguments. Other TTS adapters and `tts.load()` remain source-compatible.

The internal segment/request boundary must not assume it owns an entire document, leaving a later sentence scheduler able to chain independent iterators. This change does not implement that scheduler.

## Memory model and failure behavior

For BF16 K/V, maximum DiT cache bytes are:

```text
2 x NFE x layers x branches x heads x (bucket x 5) x head_dim x 2 bytes
```

At 512 patches this is approximately 0.70 GiB for MeanFlow and 3.52 GiB for SOAR. Qwen, semantic, vocoder state, weights, and transient attention workspaces are additional and remain subject to the 30 GiB measured gate.

The runtime rejects patch budgets above 512, invalid merge sizes, cache overflow, and inconsistent request state. It never silently falls back to full-history DiT execution. Patch-boundary evaluation materializes cache mutations so lazy graphs cannot retain all prior work.

Cache K/V remains in activation precision even for int8 weight artifacts. Cache quantization is a separate quality/performance decision and is outside this change.
