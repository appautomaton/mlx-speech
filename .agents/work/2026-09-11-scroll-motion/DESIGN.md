# Scroll-driven presentation

The hero combines a decorative SVG signal with scroll motion across its title,
supporting copy, and console. Three signal traces change amplitude while a
highlight travels along their path. This illustration is separate from the
recording's waveform and playback state.

A small independent script maps normal scroll position to one CSS variable.
Title lines separate horizontally and rise, supporting text rises more slowly,
and the console recedes and rotates slightly while moving down. Reserve space
below the hero so the console does not cover the following section. Section
headings retain their native CSS entry timelines.

Reduce travel on stacked layouts. Each entering element settles before reading
or interaction; keyboard focus removes motion from the relevant console.
The motion button pauses SVG animation and removes scroll transforms. Persist
this choice where storage is available, and always honor reduced motion. Stop
the decorative animation when the hero is offscreen or the document is hidden.
No animation dependency or media hooks. Without scripts, the SVG stays still.

Navigation uses Bricolage Grotesque at 700 weight. The theme control switches
from the fine monospace label face to Instrument Sans at 650 weight. Keep IBM
Plex Mono for the console's technical labels and code.
