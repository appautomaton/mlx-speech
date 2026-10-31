---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: Edge0/Audio8-ASR-Infinite
tags:
- mlx
- apple-silicon
- automatic-speech-recognition
- streaming-asr
- speech-recognition
- transcription
---

# Audio8-ASR-Infinite — MLX (bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![int8 build](https://img.shields.io/badge/variant-int8%20(default)-1f6feb)](https://huggingface.co/appautomaton/audio8-asr-infinite-int8-mlx)

Pure-MLX **bf16** conversion of Edge0's
[Audio8-ASR-Infinite](https://huggingface.co/Edge0/Audio8-ASR-Infinite) for
**live, unlimited-length** transcription of Chinese and English on Apple
Silicon. The complete waveform-to-text runtime uses
[`mlx-speech`](https://github.com/appautomaton/mlx-speech); PyTorch and
Transformers are not imported during inference. Weights ship as plain
`.safetensors`.

> This is the **unquantized reference** build. Most users should prefer the
> [**int8 build**](https://huggingface.co/appautomaton/audio8-asr-infinite-int8-mlx):
> it produced identical transcripts in local checks at half the size. Use bf16
> when you want the full-precision baseline, for example to compare quantization
> quality yourself.

## Model details

- Upstream developer: Edge0 ([model](https://huggingface.co/Edge0/Audio8-ASR-Infinite) · [code](https://github.com/Edge0-AI/Audio8-ASR-Infinite))
- MLX conversion and runtime: [App Automaton](https://github.com/appautomaton/mlx-speech)
- Task: streaming automatic speech recognition, Chinese and English
- Architecture: causal Voxtral-Realtime audio encoder, audio projector, and a
  delay-conditioned Qwen2.5-3B text decoder that emits one token per audio
  step; about 4B parameters
- Precision: bf16, the upstream values. The converter transposes the two
  Conv1d weights into MLX layout and merges the semantic VAD heads into the
  main weight file.
- Context: 30 s native. A rolling window keeps memory and per-step cost
  constant for audio of any length.
- Input: mono audio, resampled to 16 kHz by the public file adapter; live
  sessions take 16 kHz samples
- Runtime: batch one on Apple Silicon

## Variants

| Repo | Precision | Size | Speed on M5 Max | Notes |
| --- | --- | ---: | ---: | --- |
| [`audio8-asr-infinite-int8-mlx`](https://huggingface.co/appautomaton/audio8-asr-infinite-int8-mlx) | int8 affine | 4.4 GB | 4× real time | **default**, recommended for most users |
| `audio8-asr-infinite-bf16-mlx` (this repo) | bf16 | 8.2 GB | 3× real time | unquantized reference |

## Usage

```bash
pip install mlx-speech
```

```python
import mlx_speech

asr = mlx_speech.asr.load("audio8-asr-infinite-bf16")
result = asr.generate("speech.wav", language="zh")  # "zh" or "en", required
print(result.text)
```

```bash
mlx-speech asr \
  --model audio8-asr-infinite-bf16 \
  --audio speech.wav \
  --language zh
```

For a live waveform stream:

```python
session = asr.stream_session(language="en", frame_len=4, target_delay_ms=480)
for pcm_chunk in microphone:
    emitted_token_ids = session.feed(pcm_chunk)
    consume(emitted_token_ids)
consume(session.finalize())
print(session.result().text)
```

`feed()` accepts arbitrary sample counts and returns the token ids decoded so
far. `finalize()` is required to flush the text that still trails the audio.

## Latency modes

The decoder makes one text decision per audio step, the clock.
`target_delay_ms` sets how far the text trails the audio. The checkpoint was
post-trained for these settings:

| `frame_len` | Clock | `target_delay_ms` |
| ---: | ---: | --- |
| 4 (default) | 80 ms | 240, 320, **480 (default)**, 560 |
| 6 | 120 ms | 240, 480 |
| 8 | 160 ms | 320, 480 |

Other multiples of the clock run but were not post-trained. PCM packet size is
independent of these settings and does not change the emitted tokens.

## Release validation

bf16 and int8 transcribed two synthetic paragraphs generated with Fish S2 Pro.
Both are longer than the 30 s native context, so the rolling window was active.
The two builds produced identical transcripts.

| Language | Audio | bf16 | int8 |
| --- | ---: | ---: | ---: |
| Chinese | 36.9 s | 1.24% CER | 1.24% CER |
| English | 41.7 s | 0.75% WER | 0.75% WER |

Both Chinese errors are homophones (再→在, 她→他); the English error is one
article. Synthetic speech establishes the local release gate, not real-world
accuracy.

## Limitations

- Chinese and English only, and the language must be set explicitly.
- Greedy decoding at batch size one; no timestamps.
- Live sessions return token ids. A byte-level token can split a rare Chinese
  character, so decode the accumulated ids rather than each id alone.
- Semantic VAD probabilities are exposed per step but not used for endpointing.
- Inference requires Apple Silicon and MLX.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Runtime guide: [`docs/audio8-asr-infinite.md`](https://github.com/appautomaton/mlx-speech/blob/main/docs/audio8-asr-infinite.md)
- Project page: [appautomaton.com/mlx-speech](https://appautomaton.com/mlx-speech/)
- Default variant: [`appautomaton/audio8-asr-infinite-int8-mlx`](https://huggingface.co/appautomaton/audio8-asr-infinite-int8-mlx)
- Upstream model: [`Edge0/Audio8-ASR-Infinite`](https://huggingface.co/Edge0/Audio8-ASR-Infinite)

## License

Apache 2.0, following the upstream Edge0 release. This is an MLX conversion of
[`Edge0/Audio8-ASR-Infinite`](https://huggingface.co/Edge0/Audio8-ASR-Infinite);
refer to the upstream model card for current terms.
