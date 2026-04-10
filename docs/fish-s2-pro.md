# Fish S2 Pro

## What It Is

Fish S2 Pro is a 4B parameter dual-autoregressive TTS model from Fish Audio. It generates speech from text with inline emotion and style control via free-form tags, and supports zero-shot voice cloning from a short reference audio clip.

## Key Properties

| Property | Value |
| --- | --- |
| Architecture | Dual-AR (Slow AR 36 layers + Fast AR 4 layers) |
| Sample rate | 44,100 Hz |
| Codec | Modified DAC — 10 codebooks (1 semantic + 9 residual) |
| Frame rate | ~21.5 Hz |
| Max sequence | 32,768 tokens |
| Quantization | int8 (4.5 GB) |

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("fish-s2-pro")
result = model.generate("Hello from Fish S2 Pro!")
# result.waveform, result.sample_rate
```

```bash
mlx-speech-tts --model fish-s2-pro --text "Hello!" -o output.wav
```

## Emotion and Style Tags

Fish S2 Pro supports free-form natural language tags in `[bracket]` syntax, placed inline with the text. These are not special tokens — the model was trained on 15,000+ unique tags and interprets them as prosody/style instructions.

**Common tags:**

| Category | Tags |
| --- | --- |
| Emotion | `[excited]` `[angry]` `[sad]` `[surprised]` `[shocked]` `[delight]` |
| Voice | `[whisper]` `[low voice]` `[shouting]` `[screaming]` `[loud]` `[low volume]` |
| Expression | `[laughing]` `[chuckle]` `[sigh]` `[inhale]` `[exhale]` `[panting]` `[tsk]` |
| Pacing | `[pause]` `[short pause]` `[emphasis]` |
| Style | `[singing]` `[excited tone]` `[laughing tone]` `[professional broadcast tone]` |
| Volume | `[volume up]` `[volume down]` `[echo]` |

**Usage:** Place tags anywhere inline. They affect the speech that follows them.

```python
result = model.generate(
    "So I was walking down the street. [excited] And then I saw it! "
    "[laughing] You won't believe this. [whisper] It was a tiny hat."
)
```

Tags can be arbitrarily descriptive — the model handles free-form text like `[whisper in small voice]` or `[pitch up]` as well as the common short forms.

## Voice Cloning

Provide a reference audio file (WAV, any length) and its transcript to clone a voice.

```python
result = model.generate(
    "[excited] I tell you what, that boy ain't right!",
    reference_audio="reference.wav",
    reference_text="Transcript of the reference audio goes here.",
)
```

```bash
mlx-speech-tts --model fish-s2-pro \
  --text "[excited] I tell you what!" \
  --reference-audio reference.wav \
  --reference-text "Transcript of the reference audio." \
  -o cloned.wav
```

Both `reference_audio` and `reference_text` must be provided together. The transcript should closely match what is spoken in the reference audio.

## Generation Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `max_new_tokens` | 1024 | Controls maximum audio length. ~47 seconds at 21.5 Hz. Increase for longer text. |
| `temperature` | 0.8 | Sampling temperature (model default, not adjustable via public API) |
| `top_p` | 0.8 | Nucleus sampling threshold |
| `top_k` | 30 | Top-K token filter |

For long scripts with many emotion tags, use `max_new_tokens=2048` or higher. The model generates roughly 2 tokens per character of text, plus extra frames for tags like `[pause]`, `[laughing]`, etc.

## Token Budget

Emotion tags consume audio tokens beyond the text content:

- Pure text: ~1.0 tokens per character
- Text with tags: ~1.8-2.0 tokens per character
- Tags like `[pause]` and `[laughing]` produce audio frames with no corresponding text

If generation cuts off mid-sentence, increase `max_new_tokens`.

## Performance

- KV cache enabled — O(n) generation, not O(n^2)
- Fast AR compiled via `mx.compile` for reduced dispatch overhead
- Precomputed semantic logit bias and token mappings
- ~21 tokens/second on Apple Silicon with int8 weights

## HuggingFace

- Alias: `fish-s2-pro`
- Repo: [appautomaton/fishaudio-s2-pro-8bit-mlx](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx)

## Script

The low-level script offers additional post-processing flags:

```bash
python scripts/generate_fish_s2_pro.py \
  --text "[whisper] Hello." \
  --model-dir models/fish_s2_pro/mlx-int8 \
  --output outputs/sample.wav \
  --max-new-tokens 1024 \
  --trim-leading-silence \
  --normalize-peak 0.95
```
