# Engineering Review Sections

Use this as a trigger-based risk checklist. Evaluate a section only when PLAN.md, DESIGN.md, or the changed surface touches its subject, and do not write "No issues found" filler: summarize clean sections in one line at most and expand only verdict-driving risks. A section earns its space by naming a failure mode before execution pays for it.

## Sections

Architecture, Errors and rescue, Security, Data flow and edge cases, Code quality, Tests, Performance, Observability, Deployment, Design and UX, and Trajectory.

The list is closed so the review stays bounded. Ten of the eleven trigger on their own subject. Trajectory is the exception, because it asks a question the changed surface does not raise on its own: what does this plan make expensive to undo? It triggers on an architectural commitment, a new dependency, a durable schema or API, or a path-dependent abstraction.

Bound each section to the surface the plan actually touches. Rate security findings by likelihood and impact with mitigation status, since that is the one section where a low-likelihood finding can still block.

## Sharpeners

The checks above are the standard sweep. These are the ones reviews usually miss:

- **Name the specific exception.** Catch-all handling is the finding, because it hides the failure until it is expensive. Every rescued error retries, degrades, or re-raises with context.
- **An abstraction with no second caller is over-engineering.** The opposite finding, a happy path with no failure branch, is under-engineering. Both are worth naming.
- **Name the test that catches the failure mode you fear most.** A plan that lacks that test has a finding, whatever its coverage looks like.
- **Ask whether logs alone reconstruct the incident.** Metrics that say it works are common. Evidence of what happened is rarer.
- **Name what breaks while old and new code run together.** Deployment findings hide in the overlap, not in either version.
- **Say what the rollback actually costs.** Every integration point needs a revert, flag, or migration rollback, and how long it takes.
