# Engineering Review Sections

Use this as a trigger-based risk checklist. Evaluate only sections whose trigger appears in PLAN.md, DESIGN.md, or the changed surface, and do not write "No issues found" filler: summarize clean sections in one line at most and expand only verdict-driving risks. A section earns its space by naming a failure mode before execution pays for it.

## Triggers

| Section | Evaluate when the plan touches |
|---|---|
| Architecture | a new pattern, cross-module integration, state machine, pipeline, external service, or unclear component boundary |
| Errors and rescue | new error handling, external calls, persistence, retries, parsing, async jobs, or user-visible failure states |
| Security | new user input, an auth or permission boundary, secrets, file paths, network calls, dependencies, sensitive data, or an injection surface |
| Data flow and edge cases | a new data transform, persistence, UI interaction, workflow state, or user-visible async behavior |
| Code quality | code organization, shared modules, or repeated patterns |
| Tests | any added or changed behavior |
| Performance | new queries, loops over unbounded data, background jobs, large files, caching, concurrency, or external calls |
| Observability | operated service behavior, production workflows, async jobs, external dependencies, or hard-to-debug state |
| Deployment | migrations, feature flags, config changes, data compatibility, or irreversible state |
| Trajectory | an architectural commitment, a new dependency, a durable schema or API, or a path-dependent abstraction |
| Design and UX | UI scope |

Bound each section to the surface the plan actually touches. Rate security findings by likelihood and impact with mitigation status, since that is the one section where a low-likelihood finding can still block.

## Sharpeners

The checks above are the standard sweep. These are the ones reviews usually miss:

- **Name the specific exception.** Catch-all handling is the finding, because it hides the failure until it is expensive. Every rescued error retries, degrades, or re-raises with context.
- **An abstraction with no second caller is over-engineering.** The opposite finding, a happy path with no failure branch, is under-engineering. Both are worth naming.
- **Name the test that catches the failure mode you fear most.** A plan that lacks that test has a finding, whatever its coverage looks like.
- **Ask whether logs alone reconstruct the incident.** Metrics that say it works are common. Evidence of what happened is rarer.
- **Name what breaks while old and new code run together.** Deployment findings hide in the overlap, not in either version.
- **Say what the rollback actually costs.** Every integration point needs a revert, flag, or migration rollback, and how long it takes.
