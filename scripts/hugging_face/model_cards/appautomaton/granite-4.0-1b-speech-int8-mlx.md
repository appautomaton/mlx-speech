---
language:
- en
- fr
- de
- es
- pt
- ja
license: apache-2.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: ibm-granite/granite-4.0-1b-speech
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- automatic-speech-recognition
- multilingual
- granite
- quantized
- int8
- 8bit
---

# Granite Speech 4.0 1B — MLX (int8)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.renocrypt.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

Pure-MLX selective-int8 conversion of IBM's
[`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)
for local automatic speech recognition on Apple Silicon. The complete
waveform-to-text runtime uses `mlx-speech`; PyTorch, Transformers, `mlx-audio`,
vLLM, and cloud APIs are not runtime dependencies.

## Model details

- Upstream developer: IBM
- MLX conversion and runtime: [App Automaton](https://github.com/appautomaton/mlx-speech)
- Task: offline greedy automatic speech recognition
- Supported upstream speech inputs: English, French, German, Spanish,
  Portuguese, and Japanese
- Input: mono audio, resampled to 16 kHz by the public file adapter
- Runtime: Apple Silicon with MLX

## Precision

The Granite causal language model uses 8-bit affine weight quantization with
group size 64. The acoustic encoder, QFormer projector, normalization,
convolution, biases, activations, and KV cache retain BF16 precision. The exact
quantization contract is stored in `config.json`, and the runtime reconstructs
the saved quantized module set from checkpoint `.scales` tensors before strict
weight loading.

## Release validation

On an Apple M5 Max, isolated BF16/int8 checks used two 12–13 second English
samples, one excluded warmup, and five measured requests per sample.

| Metric | BF16 | Selective int8 | Change |
| --- | ---: | ---: | ---: |
| Weight bytes | 4,626,527,776 | 2,904,308,838 | -37.225% |
| Loaded MLX active bytes | 4,626,778,828 | 2,904,558,284 | -37.223% |
| Hank median inference | 0.833 s | 0.734 s | -11.823% |
| Peggy median inference | 0.825 s | 0.689 s | -16.428% |

The int8 transcripts differed from BF16 by one normalized word on each sample.
Hank WER was unchanged at 10.714%; Peggy WER changed from 3.571% to 7.143%, one
additional acronym error. On IBM's bundled multilingual sample, the first 64
tokens differed only at the first word and both retained the required
`timothy was a spoiled cat` phrase. These checks establish the local release
gate, not universal accuracy parity across all supported languages.

The released weight SHA-256 is
`cf355a69e931ccac95d5cf942c3d540ba2456f06ad89c379c8132875b9098e6c`.
It is byte-identical to the existing
[`mlx-community/granite-4.0-1b-speech-8bit`](https://huggingface.co/mlx-community/granite-4.0-1b-speech-8bit)
weight file. Full local methodology and memory results are recorded in the
[`mlx-speech` quantization gate](https://github.com/appautomaton/mlx-speech/blob/main/docs/benchmarks/granite-speech-int8-quant-gate-2026-08-03.md).

## Usage

```bash
pip install "mlx-speech>=0.5.1"
```

```python
import mlx_speech

asr = mlx_speech.asr.load("granite-speech-4.0-1b")
result = asr.generate("speech.wav", max_new_tokens=200)
print(result.text)
```

```bash
mlx-speech asr \
  --model granite-speech-4.0-1b \
  --audio speech.wav
```

For an explicit precision selector, `granite-speech-4.0-1b-int8` resolves to
the same repository. A downloaded repository can also be loaded directly by
local path.

## Limitations

- The `mlx-speech` runtime currently exposes greedy transcription only.
- Mandarin speech transcription is not an upstream capability. IBM describes
  English-to-Mandarin speech translation, which is a different task and is not
  exposed by this runtime.
- No timestamps, beam search, or native long-audio chunking are included.
- Inference requires Apple Silicon and MLX.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Runtime guide: [`docs/granite-speech-asr.md`](https://github.com/appautomaton/mlx-speech/blob/main/docs/granite-speech-asr.md)
- Project page: [appautomaton.renocrypt.com/mlx-speech](https://appautomaton.renocrypt.com/mlx-speech/)
- Upstream model: [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)

## License

Apache 2.0, following the upstream IBM Granite Speech release. Refer to the
upstream model card for current terms and intended-use guidance.
