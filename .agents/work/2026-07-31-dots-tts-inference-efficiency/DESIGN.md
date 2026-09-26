# dots.tts Inference Efficiency Design

## Decision 1: One acoustic producer, two decode sinks

Prompt preparation, schedule traversal, EOS handling, DiT sampling, semantic feedback, Qwen state, request RNG, and payload accounting remain one internal latent-patch producer.

```text
text/reference -> request-local latent patch producer
                                  |              |
                                  |              +-> bounded 1/1/4 stream decoder
                                  +-> one-shot batch decoder
```

Ordinary `generate()` drains latent patches into one AudioVAE batch decode. `generate_stream()` feeds the same patches into the existing bounded SLSTM/BigVGAN state. This intentionally reverses the parked requirement that ordinary generation drain waveform chunks: code reuse belongs above the decode sink, where it does not impose streaming overhead on non-streaming callers.

The two sinks must agree on payload patch count, sample count, seed trajectory, and waveform within the established BF16 seam tolerances. Only the streaming sink owns merge cadence, lookahead flush, and early-close semantics.

## Decision 2: Immutable execution helpers, request-owned mutable state

The loaded generator/model may retain:

- compiled pure-tensor functions keyed by operation, shape, dtype, mode, and branch count;
- inference-only packed projections derived from loaded weights;
- immutable rotary/mask/schedule helpers;
- a bounded prompt-feature LRU.

Each request exclusively owns RNG keys, cache offsets and storage, semantic state, recurrent state, latent buffers, and publication status. Compiled functions accept these arrays as inputs and return replacement arrays; they do not capture a request object or mutate model-global request state.

This split permits warm execution reuse without breaking interleaved same-seed determinism or early iterator close.

## Decision 3: Explicit decoder precision boundary

Checkpoint policy remains unchanged: AudioVAE encoder-side modules are FP32 and decoder-side modules are BF16. The decoder execution path follows loaded decoder dtypes rather than forcing FP32 inputs into BF16 weights.

The first implementation attempt keeps post-projection, SLSTM state, rolling decoder input, and BigVGAN in BF16. If real-checkpoint waveform/seam or fixed quality evidence fails, only SLSTM accumulation remains FP32; its output is cast once to `decoder.conv_pre.weight.dtype` before entering the rolling BigVGAN window. The broader decoder never remains FP32 merely because recurrence needs a narrow high-precision boundary.

Common four-frame and sixteen-frame latent chunks use cached compiled tensor functions. Residual final chunks compile lazily by observed shape. Flush reuses the compiled decoder-window graph and does not rerun recurrence.

## Decision 4: Progressive transactional DiT storage

`DiTSolverState` separates the public request maximum from current physical cache capacity. The first cache allocation uses bucket 64. When finalized history crosses a bucket edge, a new 128/256/512 cache is allocated, only the published prefix is copied and materialized, and the request swaps to it after successful copy. The prior cache remains valid if allocation or copy fails.

For later-patch attention, each NFE/layer uses storage after the published offset as scratch:

```text
[ published persistent prefix | fresh previous unit | current hidden/noisy tail ]
  visible before the step       publish on success    never publish
```

Fresh K/V is slice-written before attention and read through one contiguous prefix-plus-tail view. Request offsets remain unchanged until every NFE succeeds. On success they advance by one five-token finalized unit. On failure, scratch beyond the published offset is ignored and may be overwritten by the retry. This removes full-prefix concatenation without weakening delayed commit.

## Decision 5: Inference packing does not alter checkpoints

Packed QKV and adaptive-modulation projections are derived after checkpoint load and stored as inference helpers, not registered serialized parameters. Original module names, tensors, conversion output, base/int8 policy, and full-history oracle stay intact.

The generator caches solver/runner helpers by mode and execution signature; request state stays separate. A compiled helper is enabled only when its warm timing beats the equivalent eager/fused path and numerical parity passes. A compile regression is recorded and the faster correct helper remains active.

## Decision 6: Prompt reuse is bounded and content-aware

The generator holds a 256-entry memory-only LRU keyed by normalized reference-audio content and prompt mode. It stores the unscaled speaker embedding and, for continuation mode, the pre-sampling AudioVAE latent distribution. Speaker projection/scaling and random sampling remain request-local.

Audio paths are still loaded and normalized before content hashing, so file changes cannot return stale features. Cache lookup and insertion are lock-protected; expensive feature computation occurs outside the lock, and duplicate concurrent computation is acceptable. No audio or derived state is serialized.

## Decision 7: Performance evidence is local and cached-only

The profiler records a source digest, artifact identity, MLX version, warmup/compile time, cold and repeated-reference TTFC, stage totals, RTF, output health, and peak memory. Raw JSON under gitignored `outputs/dots_tts/inference_efficiency/` remains diagnostic only. A compact, machine-readable comparison contract in `slices/slice-001.md` preserves every performance and quality field needed by the final gates, along with the identities that prove the workload, host, artifacts, reference, corpus, and evaluator are compatible. The profiler and quality runner consume that canonical contract directly and fail closed on incomplete or mismatched evidence.

The baseline and final gate each contain two cached cases per MF/SOAR variant:

- `batch`: ordinary synthesis total time and RTF; this is the 35% completion gate;
- `stream`: time to first non-empty waveform chunk plus streaming completion time; TTFC must remain responsive but is not substituted for the batch-total gate.

Each case uses one compilation warmup, then clears only the prompt-feature cache before three measured requests. Request one records cold reference preparation; requests two and three record same-reference warm behavior. Uncached execution is used only by focused numerical oracle tests. No performance Markdown report or model-card claim is produced.
