# mlx-speech landing page

Static GitHub Pages site for [mlx-speech](https://github.com/appautomaton/mlx-speech),
published at <https://appautomaton.github.io/mlx-speech/>.

## Stack

A single self-contained `index.html` — no build step, no framework. Styles and
the small bit of JS (theme toggle, mobile menu, copy, tabs, scroll-reveal,
equalizer, model rendering) live inline.

- **Type:** Bricolage Grotesque (display), Hanken Grotesk (body), JetBrains Mono
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

## Regenerating the OG image

`assets/og.png` (1200×630) is rendered from a small HTML card with headless
Chrome so the real fonts are used. See git history for the `_og.html` template,
or screenshot any 1200×630 card that matches the hero.
