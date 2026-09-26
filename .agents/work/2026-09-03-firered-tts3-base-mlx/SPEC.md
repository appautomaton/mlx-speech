**Bet:** FireRedTTS3 Base can run as a practical end-to-end MLX voice-cloning pipeline on Apple Silicon without paid services or a PyTorch runtime.

# FireRedTTS3 Base MLX

## Bounded Goal

Bring FireRedTTS3 Base into `mlx-speech` as a pure-MLX, local-path-first model that accepts target text, a reference transcript, and reference audio, then produces usable 24 kHz cloned speech from a local BF16 checkpoint.

## Classification

- **Work scale:** capability
- **Work shape:** feature
- **Selected lenses:** product, engineering, runtime

## Target User

Apple Silicon developers who want to run FireRedTTS3 Base voice cloning locally through the existing `mlx-speech` API and CLI.

## Scope Coverage

### Included

- FireRedTTS3 Base zero-shot voice cloning with an explicit language, reference transcript, and reference waveform.
- The complete MLX inference path required to reach waveform output: text tokenization, RedAE, CAM++, patch encoding, Qwen3 autoregression, DiT flow generation, stopping, and waveform trimming.
- A torch-free conversion path from the pinned official FP32 checkpoints to `models/firered/firered_tts3/mlx-bf16/`.
- BF16 floating-point weights. Numerically sensitive operations may compute in FP32 when needed, without retaining a second FP32 weight copy.
- Local-path loading through the unified TTS API and CLI, focused tests, one real local end-to-end check, and concise usage documentation.
- Direct local reference inference with the pinned official source and original
  weights under the existing `.venv-torch`, using the MPS backend. Only the
  minimum dev-only compatibility shim needed to replace CUDA/FlashAttention
  assumptions is included.

### Deferred / Not in Scope

- FireRedTTS3 Instruct, voice design, semantic editing, and acoustic editing.
- Quantized artifacts and streaming generation.
- Hugging Face publication or a public model alias.
- An exhaustive golden-fixture or benchmark program. A small fixture is allowed only when it directly protects a difficult inference boundary.
- A second virtual environment, parity service, or generalized oracle framework.
- WeText, fastText, Faster Whisper, external LLM normalization, cloud GPUs, hosted inference, or paid APIs.

## Approved Approach

Implement Base first with BF16 weights. The official Qwen3 path already selects BF16 compute, and BF16 has a safer exponent range than FP16 for this transformer and flow stack. Run the official Base pipeline directly from `.references/FireRedTTS3/` with the original weights in the existing `.venv-torch`. Use MPS and add only the small dev-only compatibility shim required by the source's hardcoded CUDA device and FlashAttention configuration. PyTorch remains the local reference, not part of the MLX package.

## Reference Sources

- Official source: `.references/FireRedTTS3/` at commit `1d32ba780da6af37a71bdfd9c68c12003e908a46`, read-only.
- Official weights: `models/firered/firered_tts3/original/` at Hugging Face revision `dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610`, gitignored.
- Reference code and original weights remain development inputs only. Neither may become a runtime dependency or be mixed into the converted artifact.

## Required Outcome

- `mlx_speech.tts.load()` recognizes a local FireRedTTS3 BF16 directory and returns an adapter implementing the existing common TTS contract.
- The adapter accepts the cloning inputs without exposing upstream PyTorch objects or adding a FireRed-specific object to the shared protocol.
- Generation runs entirely with MLX and returns a finite, non-silent 24 kHz waveform that speaks the requested text in the reference voice.
- Conversion reads the official safetensors and CAM++ checkpoint without Torch, writes a separate BF16 artifact, and preserves the tokenizer and configuration assets required at runtime.
- The official Base pipeline produces one local MPS reference waveform from the same prompt, text, seed, and original weights used for the MLX check.
- Tests cover the model-loading contract and the inference logic most likely to break. Small oracle fixtures are added only where a direct PyTorch-MPS comparison usefully protects a difficult boundary.

## Constraints and Risks

- Runtime and conversion must not import Torch, torchaudio, Transformers, `mlx-lm`, `mlx-audio`, FlashAttention, or the upstream package. The dev-only MPS reference command is the explicit exception and must remain outside `src/`.
- No implementation or verification step may require a paid service, cloud GPU, hosted model, or external text-normalization API.
- Original and converted weights stay gitignored and separate from `.references/`; no model weights enter Git history.
- The complete Base path is about 3.07 billion parameters. BF16 reduces storage, but peak unified memory still needs a real local check.
- The official code hardcodes CUDA, while `.venv-torch` has MPS but no FlashAttention. The reference command must make only the device and attention substitutions needed for inference; it must not become a maintained PyTorch port.
- RedAE waveform reconstruction and CAM++ audio preprocessing must be correct enough for intelligible speech and recognizable speaker conditioning; shape-only success is insufficient.

## Acceptance Criteria

1. A torch-free command reproducibly converts the pinned original Base, RedAE, CAM++, tokenizer, and configs into a separate `models/firered/firered_tts3/mlx-bf16/` directory and rejects incomplete input.
2. A dev-only command uses the existing `.venv-torch`, pinned official source, original weights, and PyTorch MPS to produce a 24 kHz reference waveform without any paid or remote service.
3. Loading and generation through the public TTS API import no banned runtime dependency and execute model computation only with MLX.
4. Given valid target text, language, reference transcript, and reference audio, the local BF16 model produces a finite, non-silent waveform with sample rate `24000`.
5. One checked local MLX generation is intelligible as the requested text and audibly conditioned on the reference speaker; its invocation uses the same inputs as the local PyTorch-MPS reference.
6. Invalid reference inputs, unsupported language tags, and malformed or incomplete checkpoints fail with actionable errors.
7. Focused tests cover conversion alignment, loading, prompt construction, generation stopping, deterministic seeding, and waveform trimming; `pytest tests/unit/` passes.
8. The model guide documents local conversion and usage, BF16 precision, supported inputs, upstream revisions, known limitations, and voice-cloning consent requirements.

## Anti-Goals

- Do not wrap, invoke, or silently fall back to the official PyTorch runtime.
- Do not create another virtual environment or turn the small MPS compatibility shim into a second maintained inference implementation.
- Do not require cloud infrastructure, paid inference, or an external API.
- Do not vendor upstream code into `src/` or put model weights in Git.
- Do not call latent-only or token-only generation a completed TTS pipeline.
- Do not implement Instruct, quantization, streaming, publication, or extensive benchmarking in this change.
