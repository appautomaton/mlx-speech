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

Status: complete. Includes the animated SVG, independent title/copy/console
movement, pause control, and stronger navigation typography.

## Verification

Reviewed desktop and 390px screenshots and a scroll sequence from 0 to 400px.
At scroll position 300 on the 1152px viewport, the title rises 52.9px, supporting
copy rises 20.2px, and the console moves down 81.7px with scale and rotation.
The SVG's amplitude and traveling highlight change continuously at rest.
The console remains 144px clear of the next section at that scroll position.

At 390px, checks from scroll position 0 through 1300 found no horizontal overflow
or overlapping copy/console/section boundaries. Pause removes parallax and SVG
animation; reload preserves this choice and Resume restores motion. All browser
checks reused the same preview tab. Audio remained paused with zero media requests.

An isolated event harness verified scroll-frame batching, pause/resume,
remembered preferences, live reduced-motion changes, offscreen/hidden suspension,
and denied storage. CSS also gates the transforms on the motion preference.
The source catalog, recording, playback code, and content remain unchanged.

Required unit suite: 1,121 passed in 7.42 seconds, with one pre-existing regex
escape warning. JavaScript syntax and `git diff --check` passed. The change is
for draft PR #39; production remains unchanged.
