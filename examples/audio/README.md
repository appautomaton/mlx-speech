# Audio Examples

Sample outputs generated locally on Apple Silicon using mlx-speech.

## moss_local_short.wav

Short TTS sample generated with MossTTSLocal.

```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --output outputs/moss_local_short.wav
```

## moss_local_short_continue.wav

Continuation of the short sample above using MossTTSLocal continuation mode.

```bash
python scripts/generate_moss_local.py \
  --mode continue \
  --reference-audio outputs/moss_local_short.wav \
  --text "And here is the continuation." \
  --output outputs/moss_local_short_continue.wav
```

## moss_sound_effect_thunder.wav

Sound effect generated with MOSS-SoundEffect.

```bash
python scripts/generate_moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/moss_sound_effect_thunder.wav
```

## ttsd_hank_peggy_repo_intro.wav

Two-speaker dialogue generated with MOSS-TTSD.

```bash
python scripts/generate_moss_ttsd.py \
  --text "[S1] Have you seen the new mlx-speech repo? [S2] Not yet, what is it?" \
  --output outputs/ttsd_hank_peggy_repo_intro.wav
```

## vibevoice_4speaker_convo.wav

Four-speaker conversation generated with VibeVoice. Demonstrates natural
multi-speaker turn-taking and voice variation at longer form lengths.

VibeVoice uses `Speaker N:` tags to distinguish voices. A 4-speaker prompt looks like:

```bash
python scripts/generate_vibevoice.py \
  --text "Speaker 1: Have you tried the new mlx-speech library? Speaker 2: Not yet, what does it do? Speaker 3: It runs TTS models locally on Apple Silicon. Speaker 4: No cloud, no PyTorch — sounds good to me." \
  --output outputs/vibevoice_4speaker_convo.wav
```
