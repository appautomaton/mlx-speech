# Qwen3-ForcedAligner

Qwen3-ForcedAligner-0.6B times a known transcript against its audio. Given
speech and the exact text spoken, it returns a start and end time for every
Chinese character and every English word. One forward pass does the work, with
no decoding loop, so a minute of audio aligns in about a quarter of a second.
The intended use is measuring pace and pauses in TTS output against its script.

The runtime is pure MLX. It does not route through PyTorch, Transformers, or the
upstream Qwen runtime.

## Getting the Model

No MLX build is published yet, so there is no alias. Download the upstream
checkpoint (`Qwen/Qwen3-ForcedAligner-0.6B`, validated at revision `c7cbfc2`)
into `models/qwen3_forced_aligner_0_6b/original/` and convert it:

```bash
python scripts/convert/qwen3_forced_aligner.py --quant int8
python scripts/convert/qwen3_forced_aligner.py --quant bf16
```

Pass `--input-dir` or `--output-dir` to override the defaults.

## Local Layout

```text
models/qwen3_forced_aligner_0_6b/
  original/    # upstream BF16 .safetensors (conversion input)
  mlx-bf16/    # unquantized MLX package, 1.8 GB
  mlx-int8/    # affine int8, group size 64, 1.0 GB
```

## Quick Start

```python
import mlx_speech

aligner = mlx_speech.align.load("models/qwen3_forced_aligner_0_6b/mlx-int8")
items = aligner.generate("line.wav", "而美联储自己盯的核心PCE，已经到了百分之三点一。", language="zh")

for text, start_s, end_s in items:
    print(f"{text}\t{start_s:.2f}\t{end_s:.2f}")

pauses = [(a.text, b.text, b.start_s - a.end_s) for a, b in zip(items, items[1:])]
```

`generate` returns a list of `AlignItem(text, start_s, end_s)` named tuples,
with times in seconds. Loading the family directory picks its `mlx-int8`
package.

## Inputs

- `audio` is a file path, or a 16 kHz mono float array (`numpy` or `mx.array`).
  Files are mixed down and resampled. Arrays at other rates are rejected; use
  `mlx_speech.audio.resample_audio` first. Arrays that exceed ±1 are scaled
  by their peak, as upstream does.
- Audio longer than 180 s is rejected, as upstream does. Split longer audio and
  align each part.
- `language` is required and must be exactly `"zh"` or `"en"`. Japanese and
  Korean need external word segmenters and are not supported.

## Text Units

The transcript is split exactly as upstream does it:

- Whitespace separates words. Inside a word, only letters, numbers, and
  apostrophes survive, so `It's` stays whole, `3.75%` becomes `375`, and
  `state-of-the-art` becomes `stateoftheart`.
- Every CJK character is its own unit; other runs stay whole. `核心PCE`
  becomes `核`, `心`, `PCE`.
- Punctuation produces no unit. A pause at a comma shows up as a gap between
  the units on either side.

`"zh"` and `"en"` use the same rule, so mixed Chinese and English text aligns
under either. A transcript with nothing alignable is rejected.

## Reading the Times

The model classifies each boundary into 80 ms bins, so every raw time is a
multiple of 80 ms. When predictions run backwards, upstream's repair is applied:
the longest non-decreasing run is kept, isolated outliers snap to a neighbour,
and longer runs are interpolated. Repaired times can fall between bins, and a
unit can have zero length.

Unit boundaries were checked against silences measured from the waveform: 54
pauses of at least 120 ms over seven clips. The clips were Chinese TTS lines, a
59 s Chinese TTS passage, a four-voice English TTS conversation, and two human
Chinese recordings. bf16 and int8 scored the same.

| Check | Result |
| --- | --- |
| Unit end vs. silence onset | median 30 ms, max 250 ms |
| Next unit start vs. silence end | median 30 ms, max 130 ms |
| Silences that fall in an aligner gap of 120 ms or more | 52 of 54 |
| Aligner gaps of 120 ms or more that overlap a silence | 52 of 60 |

Six of the eight extra gaps were 160 ms (two bins), so treat gaps of 160 ms or
less as noise. The other two were a 0.40 s comma pause in a human recording,
too breathy to count as silence, and a 0.32 s gap over a brief energy dip in TTS
speech. Of the two misses, one was a Latin acronym before a comma (`PCE，`),
whose end ran 240 ms into the pause. In the other, the gap landed one character
late.

## Quantization

The int8 build quantizes the audio tower, the embeddings, and the text decoder.
The 5000-class time head stays BF16. Over 1064 timestamps, quantizing the head
doubled the softmax-expected timing error to save 5 MB, while the audio tower
and embeddings added no measurable error.

| Build vs. BF16 (532 units, zh and en, up to 59 s) | Result |
| --- | --- |
| Identical timestamps | 99.9% |
| Median / max difference | 0 ms / 80 ms (one bin, once) |

The `quantization` block in `config.json` is re-applied on load, and only layers
saved with scales are quantized.

## Speed

Apple M5 Max, MLX 0.32.4, warm page cache, best of three runs:

| Audio | Prompt tokens | bf16 | int8 | int8 peak memory |
| --- | ---: | ---: | ---: | ---: |
| 10 s | 285 | 0.06 s | 0.09 s | 1.2 GB |
| 54 s | 1371 | 0.44 s | 0.44 s | 2.3 GB |
| 169 s | 4215 | 1.66 s | 1.54 s | 2.4 GB |

Loading takes about 0.2 s. Wall times drift by about a third between sessions;
peak memory does not. In a same-process comparison on the 169 s clip, the fused
single pass took 1.20 s against 2.53 s for the reference decoder path, and peak
memory fell from 4.0 GB to 2.4 GB, with 99.9% of timestamps identical (one
differed by one 80 ms class).

## Runtime Shape

- The checkpoint is a Qwen3-ASR model whose `lm_head` maps to 5000 time classes.
  It reuses the Qwen3-ASR audio tower, text decoder, feature extractor, and
  tokenizer from `mlx_speech.models.qwen3_asr`.
- The prompt is `<|audio_start|>`, the audio placeholders, `<|audio_end|>`, then
  each unit followed by `<timestamp><timestamp>`. A unit's start and end come
  from the two timestamps that follow it.
- Text is encoded with Hugging Face's Qwen2 pre-tokenizer (NFC plus Qwen's split
  regex). It differs from the ASR prompt path on apostrophes and case, such as
  `O'Neill` and `DON'T`.
- The decoder runs one full-sequence pass with no KV cache, using fused attention
  and native-dtype projections (`FAST_DECODE_COMPUTE`).
- Logits are computed only at the timestamp positions, in float32.
- `mlx_speech.asr.load` rejects aligner packages and points here.

## Current Limits

- One audio and one transcript per call; no batching or streaming.
- No hub alias or published build yet.
- Chinese and English only.
- The transcript must match the speech. The aligner places every unit even
  when the audio lacks it; it does not detect insertions or omissions.

## Reference Source

The upstream Qwen3-ASR repository (`qwen_asr/inference/qwen3_forced_aligner.py`
at `7c6daf7`) is the reference for text units, the prompt, and timestamp
repair. It is not vendored or imported.
