# Confucius4-R2T2

NetEase Youdao's real-time streaming finetune of Qwen3-ASR-1.7B. Same graph as
Qwen3-ASR; bf16 only. Weights:
[`appautomaton/confucius4-r2t2-bf16-mlx`](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx).

## Use

```python
import mlx_speech

asr = mlx_speech.asr.load("confucius4-r2t2")

# Streaming: float32 16 kHz mono PCM, any chunk length.
session = asr.stream_session(language="English")
for pcm in microphone:
    update = session.feed(pcm)
    print(update.committed)   # append-only; update.hypothesis may still change
print(session.finalize().text)

# Offline
print(asr.generate("speech.wav").text)
```

`stream_session` options: `chunk_ms=160`, `lookahead_ms=160`,
`unfixed_token_num=1`, `language=None` (auto), `context=""`. `feed` processes
every complete window before returning; `finalize` decodes the remaining tail.

## Streaming behavior

The session follows NetEase's chunk loop: prefix rollback, per-window token
budget, append-only commit. Each window reuses final mel frames, closed
8-second audio-encoder blocks, and the decoder KV prefix; it recomputes the
open audio block and the text after it. Decoder matmuls and attention run in
native bf16. Offline `generate` keeps the float32-cast Qwen3-ASR path.

## Convert

```bash
python scripts/convert/confucius4_r2t2.py   # original/ -> mlx-bf16/
```

## License

Weights: NetEase Youdao Model Use License Agreement (shipped as `LICENSE` in
the weight repo). Large-scale commercial use needs a separate license from
NetEase; high-risk uses are prohibited.
