# Studio scroll motion

Add visible parallax and scroll-driven heading movement to the existing local
audio-studio preview and draft PR #39.

## Acceptance criteria

- The hero console moves at a different speed from the surrounding copy.
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
