# Stop Discrimination

The `<STOP>` conditions in SKILL.md name what halts. This names how to tell a halt from an obstacle worth pushing through, which is the only judgment the list leaves open.

- **Trivial** obstacle (typo, lint, a test the plan already flagged as flaky, a dependency conflict the plan named with a resolution strategy): fix and continue.
- **Structural** obstacle (missing dependency, ambiguous instruction, stale plan): halt and report.
- **Unsure:** run one bounded diagnostic. If it is still structural or ambiguous after that, halt.

The discriminator is whether the obstacle changes what the slice means. A typo does not. A plan that references a renamed file does.
