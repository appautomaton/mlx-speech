"""One-pass serial/32/64 Qwen3-ASR benchmark on 128 local speech crops."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_speech.asr as asr
from mlx_speech.audio import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/qwen3_asr_1_7b/mlx-int8"))
    parser.add_argument("--audio", type=Path, default=Path("examples/audio/vibevoice_4speaker_convo.wav"))
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=448)
    args = parser.parse_args()
    waveform, sr = load_audio(args.audio, sample_rate=16000, mono=True)
    if len(waveform) < 15 * sr:
        parser.error("audio must contain at least 15 seconds of speech")
    rng = np.random.default_rng(0)
    lengths = np.linspace(2, 15, 128)
    starts = [int(rng.integers(0, len(waveform) - int(seconds * sr) + 1)) for seconds in lengths]
    clips = [waveform[start:start + int(seconds * sr)]
             for start, seconds in zip(starts, lengths, strict=True)]
    # Keep input order deliberately unrelated to duration; grouping is measured.
    order = rng.permutation(len(clips))
    clips = [clips[i] for i in order]
    model = asr.load(str(args.model))
    controls = dict(language=args.language, max_new_tokens=args.max_new_tokens)
    serial = None
    differences = []
    audio_encode_seconds = 0.0
    encode = model._runtime.model.get_audio_features

    def timed_encode(*args, **kwargs):
        nonlocal audio_encode_seconds
        started = time.perf_counter()
        features = encode(*args, **kwargs)
        mx.eval(features)
        audio_encode_seconds += time.perf_counter() - started
        return features

    model._runtime.model.get_audio_features = timed_encode
    print("Configuration | Clips/s | Wall time (s) | Peak MLX memory (GiB) | Equal to serial | Audio encode share", flush=True)
    print("--- | ---: | ---: | ---: | --- | ---:", flush=True)
    for size in (1, 32, 64):
        audio_encode_seconds = 0.0
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.synchronize()
        started = time.perf_counter()
        output = ([model.generate(clip, **controls) for clip in clips] if size == 1
                  else model.generate_batch(clips, batch_size=size, **controls))
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if size == 1:
            serial = output
        changed = sum(a.text != b.text for a, b in zip(serial, output, strict=True))
        if size != 1:
            differences.append(f"B={size}: {changed}/{len(clips)}")
        equal = changed == 0
        label = "Serial B=1" if size == 1 else f"Batch B={size}"
        print(f"{label} | {len(clips) / elapsed:.3f} | {elapsed:.3f} | "
              f"{mx.get_peak_memory() / 2**30:.3f} | {equal} | {audio_encode_seconds / elapsed:.1%}", flush=True)
    print("Batch transcripts differing from serial (informational): " + "; ".join(differences), flush=True)


if __name__ == "__main__":
    main()
