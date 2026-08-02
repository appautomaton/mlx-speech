# dots.tts MLX Design

## Context

dots.tts is a continuous autoregressive TTS system, not a token-codec wrapper. End-to-end inference spans text scheduling, a Qwen2.5 contextual trunk, continuous prompt latents, a causal semantic encoder, per-patch DiT/MeanFlow solving, CAM++ speaker conditioning, and a causal AudioVAE/BigVGAN waveform path.

Two pinned references define behavior:

- `.references/dots.tts` is the official PyTorch oracle.
- `.references/dots-tts-mlx` demonstrates MLX feasibility but depends on `mlx-lm` and uses a checkpoint contract that performs some PyTorch-layout repair during load.

The public and artifact contracts created here will outlive either reference checkout.

## Decision 1 — Model-family boundaries

Place dots.tts code under `src/mlx_speech/models/dots_tts/`, with high-level synthesis in `src/mlx_speech/generation/dots_tts.py` and the unified wrapper in `src/mlx_speech/tts/_adapters/dots_tts.py`.

The model-family package owns:

- config and text schedule;
- semantic encoder and latent IO;
- CAM++ speaker front end;
- AudioVAE/BigVGAN;
- DiT plus SOAR and MeanFlow solvers;
- strict checkpoint loading and quantization metadata;
- the composed autoregressive model state.

The generation layer owns prompt preparation, solver controls, autoregressive orchestration, stopping, and waveform results. The adapter only translates the shared TTS API.

Why: keeping orchestration out of the checkpoint/model definitions matches existing repository boundaries and prevents dots-specific controls from leaking into the shared protocol.

## Decision 2 — Shared Qwen2 implementation

Extract the existing internal Qwen2 primitives from `models/vibevoice/qwen2.py` into a family-neutral internal module and extend that module to support both token IDs and input embeddings, incremental KV cache, final hidden states, and tied embeddings. VibeVoice keeps a compatibility import and regression coverage.

Do not add `mlx-lm` or Transformers.

Why: a second private Qwen2 implementation would duplicate attention, RoPE, GQA, RMSNorm, and cache semantics; importing from the VibeVoice family would invert ownership; `mlx-lm` introduces a dependency boundary the spec rejects.

## Decision 3 — Source and converted checkpoint contracts

Official source snapshots remain byte-for-byte unchanged:

```text
models/dots_tts/{soar,mf}/original/
```

Converted artifacts are self-contained and MLX-native:

```text
models/dots_tts/{soar,mf}/{mlx-base,mlx-int8}/
  config.json
  llm_config.json
  mlx_config.json
  core.safetensors
  vocoder.safetensors
  speaker.safetensors
  latent_stats.safetensors
  tokenizer/
```

Conversion performs key remapping, MLX convolution transposes, vocoder weight-normalization folding, dtype conversion, and metadata emission once. Runtime loading validates and binds already-native tensors; it does not infer source layout from tensor shapes.

`latent_stats.pt` is a 3 KB zip/pickle containing two inline NumPy float32 arrays. The converter verifies the pinned source revision and expected structure, then uses a restricted unpickler that permits only the required NumPy reconstruction globals before writing `latent_stats.safetensors`. It never imports Torch and never accepts arbitrary pickle classes.

Why: source immutability preserves the oracle; native artifacts keep runtime logic explicit; a restricted reader avoids making one tiny metadata file a Torch dependency.

## Decision 4 — Source-faithful base, then selective int8

The correctness baseline is named `mlx-base`, not `mlx-bf16`, because its validated storage policy is intentionally mixed:

- BF16: Qwen, EOS, semantic encoder, DiT/solver projections, small conditioning projections, AudioVAE decoder, and BigVGAN;
- FP32: CAM++, AudioVAE encoder, encoder bridge, distribution projection, and latent statistics.

The split inside `vocoder.safetensors` is path-defined: `audio_encoder.*`, `enc_mi_layer.*`, and `pre_proj.*` remain FP32; `post_proj.*`, `dec_mi_layer.*`, and `decoder.*` are BF16. `speaker.safetensors` remains FP32. `core.safetensors` remains BF16. Metadata serializes these predicates, and strict loading rejects any tensor whose dtype disagrees with its path.

The required int8 predicate is affine int8, group size 64, on eligible Qwen2.5 Linear/Embedding paths with BF16 activations. Every non-selected tensor retains its `mlx-base` dtype. No additional component is reduced without component fixture parity and end-to-end quality evidence.

Why: converted-weight testing disproved blanket BF16 for CAM++ and AudioVAE reference encoding, while confirming BF16 waveform decoding. Qwen still provides the dominant storage reduction, so selective int8 preserves the useful size win without misrepresenting sensitive components.

## Decision 5 — Verification ladder

Before MLX component implementation, a repository-owned dev audit command runs the pinned official PyTorch checkout in an isolated oracle environment. That environment may install Torch and the official package, but it is separate from the project environment, conversion path, published package, and runtime. It emits only bounded numeric `.npz` fixtures plus a manifest; no source weights, reference audio, generated audio, or upstream code is copied into Git.

The fixture manifest records the official and community reference commits, SOAR/MeanFlow Hugging Face revisions, source tensor hashes, oracle dependency versions, command, seed, input construction, tensor names/shapes/dtypes, numerical tolerances, and fixture hashes. A verification command rejects fixtures whose manifest no longer matches the pinned sources. Fixture generation covers text scheduling, Qwen hidden/cache/EOS, latent IO, semantic prefill/decode, speaker fbank/embedding, AudioVAE encode/decode, SOAR/MeanFlow DiT steps, and both solvers.

Correctness advances in this order:

1. source manifest and tensor inventory;
2. generated and provenance-verified deterministic component fixtures against the official oracle;
3. strict `mlx-base` mixed-dtype load and component checkpoint tests;
4. `mlx-base` end-to-end waveform generation;
5. int8 strict load and end-to-end generation;
6. fixed-corpus WER, speaker-similarity, size, and memory gates;
7. local release-layout smoke;
8. remote Hugging Face smoke.

No downstream gate substitutes for an earlier one. Generated evaluation audio and weights remain gitignored; checked-in reports contain metrics and provenance only.

## Decision 6 — One Hugging Face family repository

Publish one authoritative model card at `appautomaton/dots-tts-mlx` and four independently loadable runtime subdirectories:

```text
soar/mlx-base/
soar/mlx-int8/
mf/mlx-base/
mf/mlx-int8/
```

Upload targets address only these converted directories and the card. They never upload the local family root, so `original/` cannot be included accidentally.

The internal hub alias record gains an optional artifact subdirectory. All four dots aliases point to the same repository plus an explicit relative path (`soar/mlx-base`, `soar/mlx-int8`, `mf/mlx-base`, or `mf/mlx-int8`). Existing aliases without a subdirectory retain their current behavior; the resolver does not guess a dots variant from directory names.

For remote aliases, the resolver passes an exact subdirectory allow-list to `snapshot_download` and returns that subdirectory. A clean-cache test must show that selecting one alias materializes its configs/tokenizer/weights and the root README only, with no safetensors from the other three variants. Local direct artifact paths continue to resolve without Hub logic.

Why: one family card keeps architecture, license, limitations, quantization policy, and comparisons consistent while the subdirectories retain precise variant selection.

## Decision 7 — First public surface stays non-streaming

The adapter implements the current `TTSModel.generate(...) -> TTSOutput` contract. Internal cache and incremental-patch APIs may exist for efficient autoregression, but no dots-only public streaming API is introduced.

Why: streaming requires a cross-family protocol decision and is explicitly deferred by the spec.

## Rejected Alternatives

- Import or vendor the community MLX runtime: rejects repository ownership and dependency rules.
- Use the official PyTorch package behind an MLX wrapper: not MLX inference.
- Preserve PyTorch tensor layout and transpose opportunistically at load: makes runtime loading the conversion layer.
- Publish an all-BF16 baseline: converted-weight parity disproved it for CAM++ and AudioVAE reference encoding.
- Keep the `mlx-bf16` name with FP32 exceptions: the name would misstate the artifact contract.
- Publish only int8: removes the parity and quality baseline.
- Publish four unrelated repositories: duplicates the authoritative model-card story and increases release drift.
