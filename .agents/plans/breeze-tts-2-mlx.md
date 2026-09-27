# Breeze TTS 2 — MLX inference

Status: weight package committed on `breeze-tts-2-mlx`. The text encoder
is in the library. Backbone, depth decoder, codec, and the first WAV are not.
Official weights and source are local. No Breeze synthesis has run yet.
This is the single implementation plan for this model.

Generated audio, probes, and one-off dumps go in `/tmp/breeze-tts-2-mlx/`.
Do not write them into the repository. The weight package stays gitignored.

`models/breezeblue/breeze_tts_2/mlx-bf16/` is the runtime package: 766 BF16
main tensors, including the materialized tied audio embedding, and 496 FP32
codec tensors with 58 convolution kernels in MLX layout. Mimi `codec_model`
weights are not in the package. Do not reconvert unless the original files
change.

## Goal and working rules

Generate intelligible English and Chinese speech directly in MLX on Apple
Silicon, then support voice cloning, direction, and useful waveform streaming.
Get a real WAV early. Listen to it and fix concrete problems before expanding.

- No PyTorch or torchaudio at any stage, including conversion, tests, or parity.
  Do not install upstream or community dependency sets.
- Use the official inference source for behavior and the community MLX code
  for implementation ideas. Read it critically; do not transplant a framework.
- Keep existing project dependencies and TTS interfaces. No runtime dependency
  on Transformers, qwen-tts, mlx-audio, or the reference checkouts.
- Start with original BF16 main weights and FP32 codec. Quantization is later.
- No PyTorch parity suite, repeated weight hashing, elaborate evidence system,
  or exhaustive tensor comparisons. Add focused tests where they catch a real
  risk, such as weight layout, cache positions, or codec chunk boundaries.
- Completion means the relevant feature works with real audio. Mock output,
  nonempty tensors, or community performance claims do not establish that.

## Design discipline

Keep reasoning, progress notes, and this plan concise; do not repeatedly
rediscover settled decisions. Reuse verified project primitives and keep one
shared generation path for offline and streaming output. Add abstractions
only when they remove real duplication.

Performance basics belong in the first implementation: correct KV lifetimes,
last-position logits, device-resident sampling, and limited copies/syncs.
Batch CFG branches where compatible and grow buffers amortized. Add compilation
or specialized kernels only for measured bottlenecks; keep complexity proportional
to demonstrated latency or memory benefit.

## Local references

Official source: `.references/breeze-tts/`.
Weights: `models/breezeblue/breeze_tts_2/original/`, including `audio_tokenizer/`.
Official codec/backbone definitions: `.references/qwen-tts-0.1.1/` and
`.references/transformers-4.57.3/`. Asset details: `docs/breeze-tts-2.md`.

Existing MLX ports were found through public GitHub/Hugging Face searches.
These are selected source snapshots, not installed projects. Each `SOURCE.md`
records its upstream URL, fixed revision, files, and limitations:

| Local snapshot | What to use | What not to inherit |
| --- | --- | --- |
| `.references/mlx-audio-breeze/` | Breeze graph and Qwen MLX audio codec | Current depth loop reruns the existing frame prefix and synchronizes each code |
| `.references/mlx-breeze-tts2/` | Standalone codec, checkpoint mapping, cached depth, stream cleanup | Transformers runtime tokenization, PyTorch evaluation workflow, report infrastructure |
| `.references/BreezeTTS2_Mac_Streaming/` | Per-frame depth KV reuse and fewer CPU reads | Playback/bootstrap application or unverified performance promises |
| `.references/breeze-tts-mlx/` | MLX text/backbone/depth implementations and weight mapping | Its PyTorch codec; that bridge was not downloaded |

Preserve required licenses and notices when adapting code. The original
`.references/mlx-audio/` checkout is unchanged. No community model weights
are needed for this baseline, and no community code has been run.

## 1. Load the model and prepare prompts

Put Breeze components in `src/mlx_speech/models/breeze_tts/`, orchestration in
`generation/breeze_tts.py`, and a thin adapter in `tts/_adapters/breeze_tts.py`.
Use `scripts/convert/breeze_tts.py` for any required MLX layout conversion;
write to sibling `mlx-bf16/` and keep original weights intact.

Read the effective config and source call path. The graph needs a T5Gemma2
text encoder/projection, Qwen3 backbone, depth decoder, and bundled Qwen3-TTS
reference encoder/waveform decoder. Do not assume the main config's Mimi codec
is the codec used by the public inference path.

Load safetensors with MLX, check expected keys/shapes, convert convolution
layouts, and handle tied audio/depth embeddings explicitly. Skip weights only
when the source confirms they are unused. Do not hide errors with broad
`strict=False`, guessed config defaults, or blanket dtype casts.

Use `tokenizers` and local tokenizer files. Reproduce official text segments,
`[S0]`, instruction markers, reference text/audio ordering, and special tokens.
Keep vocal-event tags intact. Distinguish text vocabulary, codec vocabulary,
backbone EOS, and codebook control IDs.

## 2. Produce and listen to the first WAV

Start with a short voice-design request and no CFG, using the real weights.
Implement the text encoder, cached backbone, depth generation, and codec
through 24 kHz mono PCM. Preserve the source's T5Gemma2 bidirectional masks,
RoPE/normalization details, and codebook offsets; the generic Qwen/Gemma names
alone do not guarantee existing project components are interchangeable.

Use backbone KV across frames and depth KV within each frame; reset depth
state for the next frame. Keep acoustic-code sampling on device rather than
reading each code to Python. Do not implement repeated full-prefix inference
as the production default. Compilation can wait until the eager MLX path works.

Save an English and a Chinese WAV. Listen for the requested words, sensible
pronunciation, and obvious distortion, silence, looping, or truncation. Debug
with small MLX/NumPy checks where useful; do not build a separate parity project.

## 3. Complete the supported voice controls

Add reference-audio encoding and exact-transcript conditioning for cloning,
then instruction-guided voice direction and the official single-CFG behavior.
Check the negative prompt branch against the source rather than guessing it.
Cover reference channel/sample-rate handling, sampling controls, EOS, seed,
and generation limits. Reject incomplete reference pairs and invalid inputs.

Listen to representative clone/design/direction outputs in both languages,
including a few vocal events. Check that cloned speech contains the target
words rather than repeating reference text. ASR can help diagnose words, but
cannot judge voice similarity or delivery. Do not claim perceptual success
without listening; user listening feedback is valid acceptance.

## 4. Integrate waveform streaming and the library API

Use `TTSOutput` and `generate` / `generate_stream`. Register Breeze family
loading for local MLX packages; a published alias is not needed yet. Expose
`instruction`, `cfg_scale`, and supported sampling controls through the adapter
without redesigning other model APIs.

Use the codec's causal streaming state so each chunk processes new codes.
Flush final partial chunks, handle EOS at an exact chunk boundary, and clean
up state when generation ends, fails, or the iterator closes. A later request
must not inherit old codec/KV state. Do not drop sound, synthesize the entire
utterance before the first yield, or silently split/truncate long input.

Listen to chunk joins and the final tail. A focused same-codes comparison of
streamed versus offline codec output is useful for boundary bugs; full-model
parity infrastructure is not required. Check cancellation and a second request.

## 5. Measure and finish

Measure first-audio latency, complete generation time, RTF, and peak memory
on this Mac, with model loading/compilation separated from warmed inference.
Compare no-CFG and CFG voice direction. Profile only where the measurements
show a bottleneck; use the downloaded cache/compilation ideas as needed.
Do not call the result real-time unless it actually keeps up with playback.

Run the existing tests relevant to modified code and add only targeted checks
for actual failure modes. Use real weights for the final waveform smoke test.
Update `docs/breeze-tts-2.md` with working API examples and known limitations.
The runtime, converter, and tests must work without PyTorch. No benchmark
reporting framework, repeated checksums, or separate serving application.
