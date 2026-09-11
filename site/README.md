# mlx-speech landing page

Static GitHub Pages site for [mlx-speech](https://github.com/appautomaton/mlx-speech),
published at <https://appautomaton.com/mlx-speech/>. No framework or build step.

## Presentation

`index.html` contains the full model catalog, transcript, and all three Python
examples. `assets/studio.css` defines the responsive audio-studio layout.
`assets/studio.js` enhances native audio playback, filters, code tabs, and copying.
Without scripts, the native player, all 15 model variants, and all examples remain
available. The recording is never autoplayed and uses `preload="none"`.

The hero's decorative SVG signal changes amplitude and carries a traveling
highlight. It is independent of the sampled waveform and audio playback.
`assets/motion.js` maps ordinary scrolling to a CSS variable: the title rises
and separates, the supporting copy moves more slowly, and the console recedes
with a slight rotation. Native CSS view timelines handle section entrances.
Navigation uses the display font at 700 weight; the theme button uses the
reading font at 650 weight instead of the fine monospace control labels.

Text stays visible throughout and scrolling is never intercepted. Stacked
layouts use smaller movements. The animation button disables decorative motion
and parallax, remembers the choice when local storage is available, and respects
system reduced-motion preferences. The signal pauses offscreen and in hidden
tabs. Keyboard focus holds console controls still. Without scripts, the SVG and
hero stay static; without view timelines, section headings stay in place.

Three curated editions each support light and dark themes: Electric (blue),
Ember (orange), and Ultraviolet (violet). `assets/appearance.js` selects the
edition before the stylesheet loads. It excludes the previous edition stored
in `sessionStorage`, so reloads change the palette when storage is available.
The theme follows the OS until the visitor uses Night mode; an explicit choice
is remembered in `localStorage`. Storage failures leave controls usable, with
an independently random palette on each load. Without scripts, Ember follows
the system color scheme. No account, cookie, or network service is involved.

Bricolage Grotesque gives display headings their character at a moderate weight;
Instrument Sans handles body text, model names, and navigation. IBM Plex Mono
is reserved for code and control labels. All three are self-hosted Latin WOFF2
files, totaling 117,032 bytes. SIL Open Font License notices are in `assets/fonts/`.
Sources: [Bricolage Grotesque](https://github.com/ateliertriay/bricolage),
[Instrument Sans](https://github.com/Instrument/instrument-sans), and
[IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono).

## Audio provenance

`assets/audio/vibevoice-conversation.mp3` is a compressed copy of the repository's
[`examples/audio/vibevoice_4speaker_convo.wav`](../examples/audio/README.md):
four speakers generated locally on Apple Silicon using VibeVoice. The page calls
it a prerecorded output; playback does not run a model in the browser.

The MP3 is 516,140 bytes (504 KiB), approximately 43 seconds, 24 kHz mono at
96 kb/s. Recreate it from the repository root with:

```bash
ffmpeg -i examples/audio/vibevoice_4speaker_convo.wav \
  -map_metadata -1 -codec:a libmp3lame -b:a 96k -ac 1 \
  site/assets/audio/vibevoice-conversation.mp3
```

The waveform uses 120 equal windows of the source WAV, with each bar proportional
to window RMS amplitude normalized to the recording's loudest window. The
transcript was generated locally with the `qwen3-asr-1.7b` adapter and is labeled
as an ASR transcription. Source precision and generation speed are not claimed.

Keep only a small curated set of web samples here. Use external asset hosting
for a large or frequently replaced audio catalog rather than accumulating
recording revisions in Git.

## Social preview

`assets/og-studio.png` is the 1200 × 630 social card, using Electric's night theme.
Its editable source is `assets/og-studio.svg`, with the same fonts and sampled
waveform as the site. Open the SVG through the local server, wait for its fonts
to load, and capture it at 1200 × 630 with device pixel ratio 1 to recreate the PNG.

## Local preview and deployment

```bash
python3 -m http.server -d site 8000
# http://localhost:8000/
```

`.github/workflows/pages.yml` publishes on pushes to `main` touching `site/`.
GitHub Pages uses GitHub Actions; `.nojekyll` bypasses Jekyll.

When updating models, keep the static headings, aliases, task counts, guides,
and published weight links consistent. Identify models requiring a GitHub
installation ahead of PyPI. Update the sitemap and WebPage `dateModified` on
substantive changes. Check all six color/theme combinations, narrow screens,
keyboard operation, and rendering without scripts. Run `pytest tests/unit/`
before publishing. Private account measurements belong outside this repository.
