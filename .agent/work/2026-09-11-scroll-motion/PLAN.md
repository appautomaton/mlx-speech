# Studio motion implementation

Execution routing: direct and serial. Parallel-safe groups: none.

### Slice 1: Add and verify scroll motion

**Objective:** Make parallax visible in the existing preview while preserving
the page's reading order, controls, and static fallback.

**Acceptance criteria:** Implement the SPEC's hero drift and entry movement.
Disable effects for reduced motion, verify desktop and narrow layouts, and keep
all content accessible. Update draft PR #39 after checks; no deployment.

**Verification:** Compare computed transforms at different scroll positions in
the existing browser tab. Check viewport overflow, reduced-motion rules, and
console keyboard focus without playing audio. Run `pytest tests/unit/` and
`git diff --check`, then review the final patch and PR wording.

Status: complete.

## Verification

The hero console moved independently by 19.5px at scroll position 300 and
44.1px at position 600 on the 1152px desktop viewport. The catalog heading
settled from a 32.5px entry offset to zero; the code console also followed its
entry timeline. Keyboard focus disabled console animation.

Desktop and 390px screenshots were reviewed. Scroll checks at 320px and 390px
found no horizontal overflow. Mobile hero drift is capped at 16px. Applying
the site's reduced-motion rule directly removed every added animation and
transform, with text opacity remaining 1. All motion is CSS and guarded by
feature and motion-preference queries; no content or media scripts changed.
The existing tab was reused and restored to the top at desktop size. Audio
remained paused with no media request during the motion checks.

Required unit suite: 1,121 passed in 7.20 seconds, with one pre-existing regex
escape warning. `git diff --check` passed. The change is for draft PR #39;
production remains unchanged.
