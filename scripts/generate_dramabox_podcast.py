"""Podcast-style multi-turn generator for DramaBox.

Builds longer, listenable clips by:
  1. Splitting each turn's body into <= ``MAX_CHUNK_S`` chunks (sentence-aware,
     quote-safe) — a pure-Python port of the upstream duration estimator +
     chunker (``.references/DramaBox/src/{duration_estimator,text_chunker}.py``),
     no torch.
  2. Re-attaching the turn's persona prefix to every chunk so the voice
     (man/woman/tone) stays consistent — DramaBox conveys speaker identity
     ENTIRELY through this natural-language prefix; there is no gender field.
  3. Generating each chunk with a fixed seed and equal-power crossfading the
     pieces together (50 ms) so chunk/turn joins are inaudible.

A "turn" is ``(persona, body)``. A single-narrator episode is one turn; a
two-host conversation is alternating turns with different personas — which is
how we demonstrate man vs. woman voices in one episode.

Run:
    .venv/bin/python scripts/generate_dramabox_podcast.py \
        --dramabox-dir models/dramabox/mlx-bf16 \
        --gemma-dir models/gemma_3_12b_it_backbone/mlx-4bit \
        --episode ep03_diffusion_dit \
        --out-dir outputs/dramabox
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_speech.generation.dramabox import DramaBoxModel

# --------------------------------------------------------------------------- #
# Duration estimation + chunking (pure-Python port of the upstream logic)
# --------------------------------------------------------------------------- #

MAX_CHUNK_S = 35.0      # hard cap per generated chunk
TARGET_CHUNK_S = 30.0   # soft cap — close the chunk before crossing this
DURATION_MULT = 1.1     # breathing-room multiplier (matches warm server)
CHARS_PER_SEC = 14.0

_PREFIX_RE = re.compile(r'^([^"\']{3,}?)(,\s*)(?=["\'])', re.DOTALL)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def estimate_speech_duration(text: str) -> float:
    """Sentence-aware duration estimate. Mirrors duration_estimator.py:100."""
    quotes = re.findall(r'"([^"]+)"', text)
    spoken = " ".join(quotes) if quotes else text
    text_len = len(spoken)
    if text_len < 40:
        cps = CHARS_PER_SEC * 0.6
    elif text_len < 80:
        cps = CHARS_PER_SEC * 0.8
    else:
        cps = CHARS_PER_SEC
    duration = text_len / cps
    duration += (spoken.count(".") + spoken.count("!") + spoken.count("?")) * 0.3
    return max(3.0, round(duration + 2.0, 1))


def _est(t: str) -> float:
    return estimate_speech_duration(t) * DURATION_MULT


def _extract_prefix(prompt: str) -> tuple[str | None, str]:
    m = _PREFIX_RE.match(prompt)
    if not m:
        return None, prompt
    return m.group(1).strip(), prompt[m.end():]


def _assemble(prefix: str | None, sentences: list[str]) -> str:
    body = " ".join(s.strip() for s in sentences if s.strip())
    if not prefix:
        return body
    if body.lstrip().startswith(("'", '"')):
        return f"{prefix}, {body}"
    return f"{prefix}. {body}"


def chunk_prompt(prompt: str) -> list[str]:
    """Split a persona-prefixed prompt into <= MAX_CHUNK_S chunks."""
    if _est(prompt) <= MAX_CHUNK_S:
        return [prompt]
    prefix, body = _extract_prefix(prompt)
    sentences = [s for s in _SENT_SPLIT.split(body.strip()) if s.strip()] or body.split()
    chunks: list[str] = []
    current: list[str] = []
    for sent in sentences:
        candidate = _assemble(prefix, current + [sent])
        if current and _est(candidate) > TARGET_CHUNK_S:
            chunks.append(_assemble(prefix, current))
            current = [sent]
        else:
            current.append(sent)
    if current:
        chunks.append(_assemble(prefix, current))
    return chunks


# --------------------------------------------------------------------------- #
# Equal-power crossfade concat (mirrors inference_server._equal_power_crossfade)
# --------------------------------------------------------------------------- #

def crossfade_concat(pieces: list[np.ndarray], sr: int, fade_ms: float = 50.0) -> np.ndarray:
    """Concatenate [C, T] float arrays with an equal-power crossfade join."""
    if not pieces:
        return np.zeros((2, 0), dtype=np.float32)
    out = pieces[0]
    fade = int(sr * fade_ms / 1000.0)
    for nxt in pieces[1:]:
        n = min(fade, out.shape[1], nxt.shape[1])
        if n <= 0:
            out = np.concatenate([out, nxt], axis=1)
            continue
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        fade_out = np.cos(t * np.pi / 2.0)   # equal-power
        fade_in = np.sin(t * np.pi / 2.0)
        head, tail = out[:, :-n], out[:, -n:]
        join = tail * fade_out + nxt[:, :n] * fade_in
        out = np.concatenate([head, join, nxt[:, n:]], axis=1)
    return out


# --------------------------------------------------------------------------- #
# Episode definitions  (persona, body)
# --------------------------------------------------------------------------- #

# Two recurring hosts demonstrate the man/woman control: the description prefix
# is the ONLY thing distinguishing the voices.
WOMAN = "A woman hosts a tech podcast, speaking warmly and clearly"
MAN = "A man co-hosts the podcast, speaking in a calm, friendly explanatory tone"

EPISODES: dict[str, list[tuple[str, str]]] = {
    # ep03 — the diffusion DiT core, as a two-host conversation (woman + man).
    "ep03_diffusion_dit": [
        (WOMAN, '"Welcome back to Deep Dive. Today we are getting to the heart of DramaBox: the diffusion transformer that actually turns noise into speech."'),
        (MAN, '"Right, and it helps to picture it. We start with pure random noise shaped like an audio latent, and over thirty small steps the model gradually removes that noise until a clean voice emerges."'),
        (WOMAN, '"That is the flow-matching idea. At each step the transformer predicts a velocity, a direction to move the latent, and we take one small Euler step along it."'),
        (MAN, '"And this is a big stack. Forty-eight transformer layers, about three point three billion parameters. Each layer attends over the audio latent and cross-attends to the text we encoded earlier."'),
        (WOMAN, '"The clever part is the timing. Position information comes from a split rotary embedding computed on the real start and end time of each audio patch, so the model always knows where in the clip it is."'),
        (MAN, '"Exactly. Next episode we will follow the latent out of the transformer and into the audio decoder and vocoder, where it finally becomes a forty-eight kilohertz waveform."'),
    ],
    # ep04 — VAE decode + dual vocoder, two-host.
    "ep04_vae_vocoder": [
        (WOMAN, '"So last time we left our clean latent sitting at the output of the diffusion transformer. Where does it go from here?"'),
        (MAN, '"First it meets the audio variational autoencoder. The decoder uses pixel normalization and causal convolutions to expand the compact latent into a rich mel spectrogram, the time-frequency picture of the sound."'),
        (WOMAN, '"And a mel spectrogram is still not audio you can play. That is where the vocoder comes in."'),
        (MAN, '"Two of them, actually. A main BigVGAN turns the mel into a sixteen kilohertz waveform, and then a bandwidth extension network upsamples that to a full forty-eight kilohertz stereo signal with its own short-time Fourier front end."'),
        (WOMAN, '"Both run in full precision to keep the high frequencies clean, and the final output is clamped to avoid clipping. And that is the whole journey: text, to latent, to mel, to waveform."'),
    ],
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="DramaBox podcast generator")
    p.add_argument("--dramabox-dir", type=Path, required=True)
    p.add_argument("--gemma-dir", type=Path, required=True)
    p.add_argument("--episode", type=str, required=True, choices=sorted(EPISODES))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/dramabox"))
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voice-man", type=Path, default=None,
                   help="Reference clip cloned for MAN turns (transcript-free voice cloning)")
    p.add_argument("--voice-woman", type=Path, default=None,
                   help="Reference clip cloned for WOMAN turns")
    p.add_argument("--max-turns", type=int, default=0,
                   help="Generate only the first N turns (0 = all)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speech pace multiplier. Pace = text / duration, so the "
                        "requested duration is divided by this; 1.2 ≈ 20%% faster.")
    args = p.parse_args()

    turns = EPISODES[args.episode]
    if args.max_turns > 0:
        turns = turns[: args.max_turns]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"dive_{args.episode}.wav"
    log_path = args.out_dir / "dive_episodes.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    t0 = time.time()
    model = DramaBoxModel.from_dir(args.dramabox_dir, gemma_dir=args.gemma_dir)
    log(f"\n[{args.episode}] model loaded in {time.time() - t0:.2f}s; {len(turns)} turn(s)")

    # gender → optional voice reference (transcript-free clone). None falls back
    # to the persona prefix's described voice (the original behavior).
    voice_map = {"woman": args.voice_woman, "man": args.voice_man, "narr": None}
    if args.voice_man or args.voice_woman:
        log(f"[{args.episode}] cloning: man={args.voice_man} woman={args.voice_woman}")

    sr = 48_000
    pieces: list[np.ndarray] = []
    total_audio = 0.0
    total_gen = 0.0
    peak = 0.0

    for ti, (persona, body) in enumerate(turns, 1):
        gender = "woman" if persona is WOMAN else ("man" if persona is MAN else "narr")
        voice_ref = voice_map.get(gender)
        full_prompt = _assemble(persona, [body])
        turn_pieces: list[np.ndarray] = []
        for ci, chunk in enumerate(chunk_prompt(full_prompt), 1):
            # Pace control: shorter requested duration → faster speech (the model
            # fills the time it's given). speed=1.2 packs the words into ~83% of
            # the estimated time.
            dur = max(3.0, min(MAX_CHUNK_S, _est(chunk) / args.speed))
            g0 = time.time()
            res = model.generate(
                chunk, duration_s=dur, cfg_scale=args.cfg_scale,
                rescale_scale="auto", steps=args.steps, seed=args.seed,
                voice_ref=voice_ref,
            )
            gen = time.time() - g0
            wf = np.array(res.waveform, copy=False).astype(np.float32)  # [2, T]
            audio_s = wf.shape[1] / sr
            peak = max(peak, float(np.max(np.abs(wf))))
            total_audio += audio_s
            total_gen += gen
            pieces.append(wf)
            turn_pieces.append(wf)
            clone = voice_ref.stem if voice_ref is not None else "no_ref"
            log(f"  turn {ti}/{len(turns)} ({gender}/{clone}) chunk {ci}: "
                f"req={dur:.1f}s audio={audio_s:.1f}s gen={gen:.1f}s RTF={gen / max(audio_s, 1e-6):.2f}x")
        # Per-turn clip — handy for auditioning each cloned voice on its own.
        turn_wf = crossfade_concat(turn_pieces, sr)
        turn_path = args.out_dir / f"dive_{args.episode}_turn{ti:02d}_{gender}.wav"
        sf.write(str(turn_path), turn_wf.T, sr, subtype="FLOAT")

    final = crossfade_concat(pieces, sr)
    sf.write(str(out_path), final.T, sr, subtype="FLOAT")
    wall = time.time() - t0
    log(f"[{args.episode}] TOTAL audio={total_audio:.1f}s gen={total_gen:.1f}s "
        f"RTF={total_gen / max(total_audio, 1e-6):.2f}x peak={peak:.3f} wall={wall:.1f}s")
    log(f"[{args.episode}] wrote {out_path} ({final.shape[1]} samples, {final.shape[1] / sr:.1f}s)")

    with log_path.open("a") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
