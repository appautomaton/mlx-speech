# Lens Selection

Default lenses: `product` + `engineering`. Add others only when justified by the change.

## Anti-Patterns

- **Include all lenses by default.** This dilutes focus. Start with the minimum and add only when justified.
- **Skip product lens for "pure engineering" changes.** Even refactoring has product implications (risk, timeline).
- **Add security as an afterthought.** If the change touches auth, data, or trust, include security from the start.
