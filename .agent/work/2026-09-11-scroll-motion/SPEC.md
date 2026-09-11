# Studio scroll motion

Add visible parallax and scroll-driven heading movement to the existing local
audio-studio preview and draft PR #39.

## Acceptance criteria

- The hero has an immediately visible animated SVG signal. Its title lines,
  supporting copy, and console have clearly different scroll speeds.
- Provide a pause control for decorative animation and parallax. Remember the
  visitor's preference and honor the system reduced-motion setting.
- Give navigation and the theme control stronger typography. Preserve this
  project's independent audio-studio identity rather than the parent site's style.
- Section headings and the quickstart console move into their resting positions
  as they enter the viewport, without hiding text or requiring interaction.
- Native scrolling and controls remain usable. Reduced-motion preferences
  disable the effects, and unsupported browsers retain the static layout.
- Desktop and narrow layouts have no new overflow or overlapping sections.
- Validate in the existing preview tab without playing audio or opening tabs.
  Run the required unit suite and update the existing draft PR.

## Anti-goals

No animation dependencies, scroll interception, delayed content loading, audio
behavior changes, production deployment, or unrelated page redesign.
