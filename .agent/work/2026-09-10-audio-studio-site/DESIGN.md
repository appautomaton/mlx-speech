# Audio studio direction

Three independent color editions frame the studio: Ember (signal orange),
Electric (cobalt blue), and Ultraviolet (violet). Each has a fully designed light
and dark theme. They are specific to MLX Speech and do not inherit the parent
site's green palette. A tightly spaced, oversized Archivo headline provides
the main identity; IBM Plex Mono labels controls and code. Both fonts are
hosted with the site and differ from the parent site's font families. Thin
rules, physical controls, and visible timing marks reference audio equipment
without pretending to be an inference app.

A small script in the head sets the palette and theme before styles paint.
Choose randomly among palettes excluding the previous session palette, using
sessionStorage when available. Default the theme to the OS preference and save
only an explicit light/dark toggle choice in localStorage. Storage failures
must not break the page. No-script rendering uses Ember and the system theme.

The hero pairs a direct statement about local speech with a real VibeVoice
recording. Its waveform derives from that recording's samples. Playback starts
only on a user action. A native audio control remains available without scripts.
The model catalog follows as compact, readable rows, with optional task filters.
The installation section contains complete copyable examples and prerequisites.

Use fluid layouts, large touch targets, visible keyboard focus, readable contrast,
and proper heading order. Do not hide content behind entrance animations.
Keep motion tied to playback and respect reduced-motion preferences. Update the
social image to match the finished page using a code-native graphic.
