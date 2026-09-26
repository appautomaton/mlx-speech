# FireRedTTS3 MLX Runtime Optimization

**Bet:** The current FireRedTTS3 Base pipeline can preserve its verified voice-cloning quality while correcting RedAE sliding attention and materially reducing MLX hot-path cost without changing the checkpoint artifact.

## Bounded goal

Optimize the existing FireRedTTS3 Base MLX inference runtime, first closing the newly discovered RedAE sliding-window parity gap and then retaining only MLX-native optimizations that measurably improve latency or memory on the existing golden request.

**Work scale:** capability

**Work shape:** parity and performance refactor

**Selected lenses:** product, engineering, runtime

## Context and baseline

The current pipeline already produces a correct 24 kHz waveform and uses a bounded Qwen3 KV cache for autoregressive continuation. On the fixed local request (`reference.wav`, exact ASR transcript, `你好，很高兴认识你。`, seed 1234, CFG 2.0, 10 flow steps), the current MLX profile is:

- RedAE encode: 0.041 seconds
- CAM++: 0.020 seconds
- Qwen3/DiT generation: 2.556 seconds
- RedAE decode: 0.048 seconds
- output: 65,280 samples at 24 kHz
- MLX peak: 5.770 GiB; process peak: 8.324 GiB
- local ASR: exact target text; reference/output CAM++ cosine: 0.7482

The core preallocates BF16 KV state for `prompt_length + max_generated_patches`, performs one full prompt prefill, and uses one-token continuation thereafter. DiT operates on the official fixed two-patch history plus current patch. Those behaviors are invariants, not gaps.

Source review found that the MLX RedAE encoder reads decoder window fields and the RedAE decoder does not enable the official 64-token sliding window. The existing attention mask also materializes a dense score matrix before masking, so its memory behavior is not window-bounded.

## Approved approach

Use the runtime-first scope: preserve the current flat artifact and focus on
parity-correct runtime changes. The profile supports this choice because
Qwen3/DiT consumes 2.556 seconds of the approximately 2.67-second inference
path, while RedAE wiring and dense masked attention expose a separate concrete
correctness and memory problem. The installed MLX runtime provides fused SDPA
and compilation primitives, but their benefit is not assumed; each candidate
must pass the repeated golden benchmark before it remains in the implementation.

## Required outcome

- RedAE encoder and decoder use their own pinned sliding-window configuration and match the official causal 64-token window semantics; CLS downsampling retains its official short full-attention behavior.
- RedAE sliding attention avoids dense full-history attention storage for long sequences. Its temporary attention memory scales with sequence length times window size, not sequence length squared.
- The existing Base Qwen3 prompt-prefill and one-token KV-cache path remains bounded by the request patch budget and continues to use grouped KV heads.
- Fixed-shape DiT, PatchEncoder, rotary/time constants, and Qwen decode are evaluated for MLX compilation or caching. An optimization lands only when the benchmark demonstrates a benefit without changing the model equations.
- Repeated generation on one loaded model does not accumulate request-owned cache or lazy graphs.
- Public API, CLI behavior, flat artifact layout, tensor names, and stored dtypes remain compatible with the existing verified artifact.

## Acceptance criteria

1. Focused wiring tests prove that RedAE encoder layers use `enc_sliding_window=64`, decoder layers use `dec_sliding_window=64`, and CLS downsampling remains full attention.
2. Numerical tests compare optimized sliding attention with the current dense-mask equation on short deterministic inputs, including positions before and after the 64-token boundary.
3. A long-sequence diagnostic demonstrates window-bounded attention allocation and finite RedAE output without relying on Torch.
4. Existing Qwen3 KV-cache tests still prove one prompt prefill followed by single-token continuation, exact cache length, budget overflow rejection, and deterministic generation.
5. Using the same machine, loaded artifact, golden request, and measurement boundaries, the retained runtime changes improve median core-generation latency or MLX peak memory by at least 10% relative to the recorded baseline. The other metric must not regress by more than 10%.
6. Two runs with the same seed produce identical MLX output within the existing deterministic tolerance; the optimized result remains finite, non-silent, mono 24 kHz.
7. Local ASR still recovers `你好，很高兴认识你。` exactly and reference/output CAM++ cosine remains at least 0.70.
8. The full unit suite, FireRed checkpoint tests, runtime tests selected by the plan, and local integration test pass without checkpoint skips.
9. FireRed runtime and conversion remain free of Torch, torchaudio, Transformers, `mlx-lm`, network calls, and new published dependencies.

## Constraints and risks

- The pinned official FireRedTTS3 source remains the equation reference; performance changes cannot alter stop logic, cosine schedule, Euler updates, CFG, two-patch history, tokenizer sequence, or prompt trimming.
- Timing is noisy. Performance evidence must distinguish cold model loading from warm inference and report repeated measurements rather than one favorable run.
- `mx.compile` may increase cache memory or recompile when shapes vary. Only bounded, stable-shape regions should be candidates, and compile cache cost counts in the memory result.
- Correcting decoder sliding attention will change the waveform relative to the currently verified but incorrectly wired decoder. Quality is judged against ASR, speaker cosine, determinism, and official equations rather than bitwise equality with the old output.
- The shared Qwen3-ASR decoder is used by another model family. Shared attention changes require its regression suite and must preserve full-attention and KV-cache behavior.

## Scope coverage

**Included:** RedAE sliding-window parity, window-bounded attention, MLX hot-path profiling, measured compile/constant-cache improvements, KV-cache regression proof, and refreshed golden integration evidence.

**Deferred:** removing the three unused RedAE token embeddings (about 779 MiB on disk), changing converter tensor counts, quantization, streaming RedAE, multi-request prompt caching, and a new published Hugging Face artifact. These require an artifact migration or a separate public behavior decision.

## Anti-goals

- Do not replace MLX execution with Torch, torchaudio, Transformers, `mlx-lm`, or a remote service.
- Do not redesign the public TTS API or introduce a FireRed-specific framework layer.
- Do not claim an optimization from code shape, synthetic microbenchmarks alone, or a single timing sample.
- Do not trade away waveform completion, deterministic seeds, ASR correctness, or speaker conditioning for benchmark gains.
- Do not change the checkpoint format or weights in this change.
