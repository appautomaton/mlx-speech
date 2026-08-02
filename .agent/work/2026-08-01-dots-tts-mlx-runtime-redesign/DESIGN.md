# dots.tts MLX-Native Inference Design

## Scope Boundary

The shipped change is the speech runtime, not its measurement tooling. The existing profiler becomes a thin local before/after timer. Candidate registries, accepted-head ledgers, trace orchestration, transaction journals, generalized evidence schemas, and public execution selectors are removed.

PyTorch source constrains model equations and observable behavior. MLX execution structure is designed independently around stable graphs, explicit request state, and bounded decoder state.

## Acoustic Execution

The generator owns one request-local acoustic state containing Qwen, semantic, DiT, RNG, and cache values. MLX functions receive mutable arrays and offsets explicitly and return candidate next state. Python publishes that state only after evaluation succeeds.

EOS uses the Qwen result already produced for the current patch. Threshold `1.0` bypasses EOS projection and scalar publication. Normal EOS materializes the decision no earlier than the current patch transaction; the patch is solved, fed back, and emitted before stopping.

DiT compilation uses a bounded set of stable signatures derived from model mode, dtype, solver shape, and physical cache capacity. Request offsets and growing logical prefixes stay tensor inputs, not compile keys. Built-in MLX normalization, rotary, vectorization, and immutable geometry are preferred when their numerical parity is explicit. A failed compilation strategy is removed rather than kept behind a selector.

## AudioVAE Bridge

The bridge processes fixed-size tiles with an explicit valid length and flattened recurrent state. Padding cannot publish hidden/cell state. Fixed-row projections use batched MLX operations rather than Python row loops. Tile shapes remain bounded and are chosen from actual batch/stream input sizes; request length does not create an unbounded compile cache.

## Stateful BigVGAN

Streaming decoder state is request-owned and bounded. It includes causal convolution tails, transpose-convolution overlap/phase, alias-free filter history, lookahead, and final-flush state. `process(new_frames, state, final)` emits only stable new samples and never reprocesses the full rolling left context.

Batch decoding uses the same correct primitives with larger tiles. Chunked and one-shot output must match within existing dtype-aware tolerances for regular, irregular, short, and final partitions. Failures do not advance state; duplicate finalization emits nothing.

Built-in MLX operations come first. A private `mx.fast.metal_kernel` is allowed only for a specific residual kernel whose equations and alignment can be tested directly and whose end-to-end effect is visible in the final default-path timing.

## Measurement Boundary

`scripts/eval/profile_dots_tts_inference.py` is reduced to a thin local timer for the real public int8 path. One invocation records one request per alias/path plus separate model-load time, wall time, streaming first-audio time, waveform duration, RTF, patch count, stop reason when available without extra synchronization, and peak MLX memory.

The timer has no candidate modes, backend/reference selectors, repetition controls, ledger, capture parser, privileged process management, or committed evidence contract. The starting invocation runs once after cleanup; the final invocation runs once after implementation. Focused microtiming may be written as a disposable test or scratch command when it decides a concrete kernel or tile choice.

## Failure and Compatibility Rules

- No compiled closure captures request RNG, mutable caches, offsets, or recurrent state.
- Cache/state publication is transactional across exceptions and iterator close.
- Batch and streaming retain current public arguments and waveform protocol.
- No speedup may come from reduced solver work, earlier stopping, truncated waveform output, changed weights, or relaxed quality gates.
- Unsupported shapes or dtypes use a correct MLX fallback; they do not select a second public backend.
