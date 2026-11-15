# Fish Audio Drama 3

## What It Is

Fish Audio Drama 3 is a multi-speaker text-to-speech model for scenes and long-form
content. One text can voice up to 16 characters, each cloned from its own reference
clip, with inline delivery tags such as `[sighs]` or `[laughs]`. It outputs 44.1 kHz
mono audio.

## Key Properties

| Property | Value |
| --- | --- |
| Backbone | Qwen3.5-MoE: 35B total, 3B active per token; 40 layers, every fourth gated attention, the rest Gated DeltaNet |
| Experts | 256 per layer, 8 routed per token, plus a gated shared expert |
| Audio decoder | 4 shared + 9 residual layers that complete the 10 codebooks of each frame |
| Codec | Fish S2 Pro codec, byte-identical: 10 codebooks, 2048 samples per frame (21.5 frames/s) |
| Weights in memory | Routed experts affine 8-bit (group 64); every other weight bf16, as released |
| Memory | About 39 GiB once loaded, about 2.5 GiB less with `dense_int8`; 44–46 GiB peak while generating |

## Weights

The upstream release loads as is; there is no conversion step and nothing needs
PyTorch:

```bash
hf download fishaudio/drama-3 --local-dir models/fish_drama_3/base
```

Its routed experts are block FP8, which MLX does not run. Loading requantizes them
in memory to MLX 8-bit, one projection at a time, in about 20 seconds; the peak is
the loaded model itself, and nothing is written to disk. Each group of 64 takes its
own minimum and maximum as its range, fitted to the exact FP8 values in fp32, which
leaves 0.57% relative error against them (`mx.quantize` on their bf16 rounding leaves
0.76%). Every other tensor is
used as released, and the codec is built from the release's `codec.pth`.

Fish Audio re-uploaded the weights on 2026-10-29 (revision `78e5ab7`), replacing the
preview weights of 2026-10-13; the measurements below are on the re-upload.

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("models/fish_drama_3/base")
result = model.generate(
    "Hello from Drama 3.",
    reference_audio="voice.wav",
    reference_text="Transcript of voice.wav.",
)
# result.waveform, result.sample_rate (44100)
```

```bash
python scripts/generate/fish_drama_3.py --text "Hello from Drama 3." \
    --reference-audio voice.wav --reference-text "Transcript of voice.wav." \
    -o output.wav
```

Without a reference the model picks a voice itself.

## Multiple Speakers

Speakers are addressed in the text with `<|speaker:0|>` to `<|speaker:15|>`.
References follow upstream fish-speech: parallel lists of audio, transcript, and an
optional speaker id. A transcript that does not start with a speaker tag gets its
speaker's tag; without an id it voices speaker 0.

```python
result = model.generate(
    "<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.",
    reference_audio=["anna.wav", "ben.wav"],
    reference_text=["Anna's transcript.", "Ben's transcript."],
    reference_speaker=[0, 1],
)
```

`model.prepare_reference(...)` encodes a voice once; pass the result back as
`reference_audio` to reuse it.

## Delivery

Without direction, a cloned voice keeps the mood and pace of its reference clip: an
angry reference reads a bedtime story angrily. A description in brackets before the
line redirects the delivery while the voice stays (technical report, Appendix C):

```python
result = model.generate(
    "[温柔轻缓，带着笑意，像在哄孩子睡觉] 从前有只小狐狸，每天晚上都爬到山顶数星星。",
    reference_audio=voice,
    reference_text="Transcript of voice.wav.",
)
```

Write the description in the language of the line. On twelve Mandarin dubbing lines,
Chinese descriptions were followed more closely than English ones; on new text read
by three voices against their reference's mood, the description changed the
delivery and kept the voice. The report also lists point tags such as `[pause]`,
`[sighs]` and `[laughs]`, span tags such as `<whisper>…</whisper>`, and director's
notes (`Direction: …` then `Line: …`); these were not compared here.

## Duration Control

`duration_seconds` asks for audio of that length, through the model's own duration
hint (`[audio tokens: N]`, technical report Table 10). The model ends within a few
frames of the request, which makes it the natural input for dubbing, where every
line has a slot:

```python
result = model.generate(line, reference_audio=voice, duration_seconds=2.8)
```

A request shorter than the text needs drops its last words. For Mandarin read at a
natural pace, allow one second per 3.3 characters, punctuation not counted: at four
per second the last words were clipped, and a voice whose reference speaks slowly
needed one second per three. A duration also generates the text in one pass and
caps it at 2.5 times the requested length.

## Stalls and How to Avoid Them

Without a duration, some takes finish a sentence and then emit silence until
`max_new_tokens`; vLLM-Omni lists this as model behaviour, more frequent after
leading fillers or tags and with pause-heavy references. In this port it showed up
mainly at sentence boundaries: single sentences completed 12 of 12 times, while two
sentences by one speaker stalled in 3 of 4 takes.

- Pass `duration_seconds`. With it, two-speaker dialogue with references completed
  4 of 4 takes, against 1 of 4 without it.
- Check each take and regenerate the ones that fail, as the technical report does.

## Generation Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `temperature` | 0.7 | upstream evaluation recipe |
| `top_p` | 0.7 | cumulative probability over the full vocabulary |
| `top_k` | 30 | |
| `max_new_tokens` | 1024 per chunk | about 47 s; 2.5 × the duration when one is given |
| `single_chunk` | False | long text is split into ~300-byte chunks of at most three speakers |
| `seed` | None | |

The main token is drawn only from `<|im_end|>` and the 4096 semantic tokens, with
upstream's repetition-aware resampling. Long text keeps the first chunk and the
latest one as context, as upstream does.

## Performance

On an M5 Max, decoding runs at about 49 frames/s, 2.3 times real time (21.5
frames/s), with a 178-row prompt prefilled in about 0.18 s. Low Power Mode halves
that to about 22 frames/s, just real time. Decoding a frame runs every layer on a
single row, so the elementwise chains are compiled into fused kernels. The codec
decodes in time tiles that reproduce a whole pass exactly, so its memory stays near
5 GB whatever the audio length. Freed codec buffers stay in MLX's cache until
`mx.clear_cache()`, so the process footprint can read 50–56 GB in between.

Everything before the current user turn (the system turn, references and history)
is prefilled apart and kept on the runtime. A later prompt that starts with the
same rows reuses it and prefills only its own turn: same reference, same history,
new line. Reuse gives the same frames as a fresh prefill, bit for bit.

`FishDrama3Runtime.from_dir(path, dense_int8=True)` packs the dense bf16
projections as affine int8 once loaded. That is
every backbone attention, Gated DeltaNet and shared-expert projection, and every
audio-decoder tower layer. The routers, the DeltaNet decay gates (`in_proj_a`,
`in_proj_b`), embeddings, `lm_head` and the codebook heads stay bf16.

Measured on 2026-11-06, one warmed process voicing one script line after another
with the same reference and history, three lines by five seeds:

| Build | RTF | Per line | Active memory |
| --- | --- | --- | --- |
| Before | 0.563 | 2.34 s | 40.5 GiB |
| Shared head reused | 0.453 | 1.86 s | 40.5 GiB |
| Shared head reused, `dense_int8` | 0.381 | 1.58 s | 38.4 GiB |

## Validation

No PyTorch parity. Every layer is checked against float64 numpy oracles written
from the upstream math. On the release, every dense tensor loads bit for bit as
released. Teacher-forced on 1143 frames of real speech in three voiced dialogues,
the int8 experts stay within 0.005 nats of exact FP8 experts on the main token and
0.002 on the residual codes (mean KL per frame).

## License

The Drama 3 weights are released under the Fish Audio Community License: free for
research and non-commercial use, limited commercial use below a revenue threshold,
and a separate agreement for anything else, including hosted services.
