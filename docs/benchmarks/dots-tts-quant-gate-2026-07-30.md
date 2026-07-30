# dots.tts Quantization Gate — 2026-07-30

**Verdict: PASS**

Affine int8 (group size 64) is applied only to eligible native Qwen Linear/Embedding modules. All other component dtypes match `mlx-base`.

## Aggregate results

| Variant | Base WER | Int8 WER | Δ WER | Base speaker cosine | Int8 speaker cosine | Δ cosine | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mf | 0.0588 | 0.0588 | +0.0000 | 0.7868 | 0.7901 | -0.0033 | PASS |
| soar | 0.0000 | 0.0000 | +0.0000 | 0.7992 | 0.8147 | -0.0155 | PASS |
| overall | 0.0294 | 0.0294 | +0.0000 | 0.7930 | 0.8024 | -0.0094 | PASS |

Thresholds: WER regression ≤ 0.0100; speaker-cosine regression ≤ 0.0200.
Mandarin uses Unicode Han characters as error-rate tokens; English uses normalized words.

## Fixed corpus

| Reference | Language | Reference text | Target text |
| --- | --- | --- | --- |
| samantha_en_us | en | My name is Samantha. I speak clearly and calmly. | Today the weather is bright and peaceful. |
| tingting_zh_cn | zh | 你好，我叫婷婷。这个声音清晰平稳。 | 今天的天气晴朗而平静。 |

## Artifact size and peak memory

| Artifact | Size GiB | Peak GiB |
| --- | ---: | ---: |
| soar/base | 4.557 | 8.308 |
| soar/int8 | 3.210 | 6.963 |
| mf/base | 4.559 | 8.521 |
| mf/int8 | 3.212 | 7.177 |

## Per-case evidence

| Artifact | Reference | Mode | Patches | Seconds | WER | Speaker cosine | ASR text |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| mf/base | samantha_en_us | continuation | 18 | 2.88 | 0.0000 | 0.7789 | Today, the weather is bright and peaceful. |
| mf/base | samantha_en_us | speaker_only | 20 | 3.20 | 0.2857 | 0.7352 | The nice. Today the weather is bright and peaceful. |
| mf/base | tingting_zh_cn | continuation | 17 | 2.72 | 0.0000 | 0.8207 | 今天的天气晴朗而平静。 |
| mf/base | tingting_zh_cn | speaker_only | 18 | 2.88 | 0.0000 | 0.8122 | 今天的天气晴朗而平静。 |
| mf/int8 | samantha_en_us | continuation | 18 | 2.88 | 0.0000 | 0.7928 | Today, the weather is bright and peaceful. |
| mf/int8 | samantha_en_us | speaker_only | 20 | 3.20 | 0.2857 | 0.7280 | The nice. Today the weather is bright and peaceful. |
| mf/int8 | tingting_zh_cn | continuation | 17 | 2.72 | 0.0000 | 0.8246 | 今天的天气晴朗而平静。 |
| mf/int8 | tingting_zh_cn | speaker_only | 18 | 2.88 | 0.0000 | 0.8150 | 今天的天气晴朗而平静。 |
| soar/base | samantha_en_us | continuation | 18 | 2.88 | 0.0000 | 0.7585 | Today, the weather is bright and peaceful. |
| soar/base | samantha_en_us | speaker_only | 19 | 3.04 | 0.0000 | 0.7875 | Today, the weather is bright and peaceful. |
| soar/base | tingting_zh_cn | continuation | 17 | 2.72 | 0.0000 | 0.8734 | 今天的天气晴朗而平静。 |
| soar/base | tingting_zh_cn | speaker_only | 17 | 2.72 | 0.0000 | 0.7772 | 今天的天气晴朗而平静。 |
| soar/int8 | samantha_en_us | continuation | 18 | 2.88 | 0.0000 | 0.7937 | Today, the weather is bright and peaceful. |
| soar/int8 | samantha_en_us | speaker_only | 19 | 3.04 | 0.0000 | 0.8005 | Today, the weather is bright and peaceful. |
| soar/int8 | tingting_zh_cn | continuation | 17 | 2.72 | 0.0000 | 0.8701 | 今天的天气晴朗而平静。 |
| soar/int8 | tingting_zh_cn | speaker_only | 17 | 2.72 | 0.0000 | 0.7945 | 今天的天气晴朗而平静。 |

## Provenance

- Corpus manifest: `examples/clone_eval/dots_tts_macos_multilingual_v1.json` (`2dcc499b3cca9130572b9cda5acc861c88a2ab57c782a2b1114c21115b335a60`)
- Corpus lock: `outputs/dots_tts/eval_corpus/manifest.lock.json` (`a77897c60a690e9a90f125a206496fd60a640a87fa886c946a1920100b04c197`)
- ASR evaluator: `models/qwen3_asr_1_7b/mlx-int8`; weights `8a9aca31c5715d080f7d891dbac08146aeddf8c34cd53e46cf24d665dcd33786`
- `soar/base` artifact digest: `abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e`; upstream revision `e3520f75254d0020a0406db31c51a79d00d22d55`
- `soar/int8` artifact digest: `0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3`; upstream revision `e3520f75254d0020a0406db31c51a79d00d22d55`
- `mf/base` artifact digest: `b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0`; upstream revision `25c53fb462e57087e52237daa5ea30df1c5cc328`
- `mf/int8` artifact digest: `424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d`; upstream revision `25c53fb462e57087e52237daa5ea30df1c5cc328`
- Host: `macOS-26.5.2-arm64-arm-64bit-Mach-O / arm64`
- Command: `scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts`
- Failed cases: none.
- Generated/reference audio and local weights are gitignored; this report contains metrics and hashes only.
- These are locally reproduced measurements, not upstream benchmark claims.
