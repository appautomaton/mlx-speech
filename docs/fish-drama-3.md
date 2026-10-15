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
| MLX checkpoint | Routed experts affine 8-bit (group 64); every other weight bf16, as released |
| Memory | About 41 GiB of weights; 44–46 GiB peak while generating |

## Converting the Release

The upstream release stores its routed experts as block FP8, which MLX does not run.
Convert it once; nothing here needs PyTorch:

```bash
hf download fishaudio/drama-3 --local-dir /tmp/drama-3
python scripts/convert/fish_drama_3.py --input-dir /tmp/drama-3 \
    --output-dir models/fish_drama_3/mlx-int8
```

The converter dequantizes each FP8 block exactly, requantizes the experts to MLX
8-bit (0.76% relative error against the FP8 values), copies every other tensor bit
for bit, and converts the codec. It holds one layer at a time, under 7 GB of memory.

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("models/fish_drama_3/mlx-int8")
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

## Duration Control

`duration_seconds` asks for audio of that length, through the model's own duration
hint (`[audio tokens: N]`, technical report Table 10). The model ends within a few
frames of the request, which makes it the natural input for dubbing, where every
line has a slot:

```python
result = model.generate(line, reference_audio=voice, duration_seconds=2.8)
```

A request shorter than the text needs drops its last words: for Mandarin, keep at
least about one second per six characters. A duration also generates the text in
one pass and caps it at 2.5 times the requested length.

## Stalls and How to Avoid Them

Without a duration, some takes finish a sentence and then emit silence until
`max_new_tokens`; vLLM-Omni lists this as model behaviour, more frequent after
leading fillers or tags and with pause-heavy references. In this port it showed up
mainly at sentence boundaries: single sentences completed 12 of 12 times, while two
sentences by one speaker stalled in 3 of 4 takes.

- Pass `duration_seconds`. With it, two-speaker dialogue with references completed
  4 of 4 takes, against 1 of 4 without it.
- Check each take and regenerate the ones that fail, as the technical report does:
  for example transcribe it with Qwen3-ASR and compare against the script.

On an 80-line Mandarin drama scene (six characters, references cut from the source
video), durations from the source timing gave 70 of 78 lines at ≥ 0.9 ASR similarity
on the first take, and 78 of 78 after regenerating the weak ones with a longer
duration or another seed.

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

About 22 frames/s on an M5 Max without competing GPU load, just above real time
(21.5 frames/s). Not yet optimized: graph compilation and asynchronous evaluation are
open. The codec decodes in time tiles that reproduce a whole pass exactly, so its
memory stays near 5 GB whatever the audio length.

## Validation

No PyTorch parity. Every layer is checked against float64 numpy oracles written
from the upstream math; on real weights, teacher-forced decoding of real speech
predicts 93% of main tokens and 55–86% of residual codes, and Qwen3-ASR transcribes
generated speech.

## License

The Drama 3 weights are released under the Fish Audio Community License: free for
research and non-commercial use, limited commercial use below a revenue threshold,
and a separate agreement for anything else, including hosted services.
