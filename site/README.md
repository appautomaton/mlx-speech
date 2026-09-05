# mlx-speech landing page

Static GitHub Pages site for [mlx-speech](https://github.com/appautomaton/mlx-speech),
published at <https://appautomaton.renocrypt.com/mlx-speech/>.

## Stack

A single self-contained `index.html` with no build step or framework. Model
cards, headings, and links are static HTML. Styles and the small bit of JS
(theme toggle, mobile menu, copy, tabs, scroll animation, equalizer) live inline.
Content remains visible without JavaScript, including both quickstart examples.

- **Type:** Big Shoulders Display (display), IBM Plex Sans (body), IBM Plex Mono
  (data) — loaded from Google Fonts via `<link>`.
- **Icons:** [Lucide](https://lucide.dev) via CDN.
- **Theme:** simplistic neutral palette + one vermilion accent. Light/dark via
  `data-theme` on `<html>`, persisted in localStorage, defaulting to the OS
  scheme. Deep-link with `?theme=light` / `?theme=dark`.
- **Responsive:** authored mobile-first; breakpoints at 760px and 900px.

Total page weight is just `index.html` + `favicon.svg` + `assets/og.png`.

## Deploy

Published by `.github/workflows/pages.yml` on every push to `main` that touches
`site/`. GitHub Pages source must be set to **GitHub Actions** (Settings → Pages).
`.nojekyll` keeps Jekyll out of the way.

## Local preview

```bash
python3 -m http.server -d site 8000
# open http://localhost:8000/
```

## Updating the page

Keep the model counts and decorative ticker in sync with the static cards.
Each card needs a model heading, loader alias, guide, and published weight link.
Mention a GitHub installation requirement when a model is ahead of PyPI.
Update `sitemap.xml` and the WebPage `dateModified` on substantive page changes.

`assets/og.png` is the 1200×630 social preview. The current revision was edited
with the built-in imagegen tool and resized for Open Graph. Preserve the dark
background, condensed headline, vermilion accent, and equalizer motif.
Keep the two supporting lines above the bars with a clear gap:

> Local TTS · voice cloning · dialogue · sound effects · ASR
> MLX-native speech for Apple Silicon.

Validate desktop and mobile layouts with JavaScript enabled and disabled.
Run `pytest tests/unit/` before publishing.
