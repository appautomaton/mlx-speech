# Frame Quality

Load this reference only before finalizing or refreshing `SPEC.md`.

## SPEC Anti-Patterns

- Vague objective: "improve", "streamline", or "harden" without observable behavior.
- Inflated importance: broad claims that do not change scope or acceptance criteria.
- Hidden assumptions: unresolved decisions smuggled into the goal.
- Solution leakage: implementation details that belong in PLAN.md unless they constrain scope.
- Missing anti-goals: no clear statement of what the change must not do.
- Scope amputation: artificially reducing a coherent capability to hit a line-count target, producing a spec that solves part of the problem and leaves the rest unframed.

## Better Shape

- State one bounded outcome in plain language.
- Convert broad goals into acceptance criteria or risks.
- Put uncertain claims under assumptions, not facts.
- Use anti-goals to prevent scope creep.
- For capability-sized goals, confirm that all acceptance criteria point at one behavioral outcome. If they test unrelated outcomes, the spec bundles independent work and should be split.

## Prose Hygiene

Specs attract significance inflation and promotional language. The bounded goal should state what changes, not why it matters to the universe.

Scan for:
- "crucial", "vital", "pivotal", "key": replace with the observable constraint or drop
- "serves as", "stands as": use "is"
- "streamline", "enhance", "leverage": name the specific operation
- Forced rule-of-three in constraints or anti-goals
- Em dashes where commas work

Before: "This crucial change serves as a pivotal step in streamlining the authentication flow — enhancing security, improving UX, and reducing latency."
After: "Add JWT validation to protected API routes. Users without a valid token get a 401."

## Final Check

If two engineers could implement materially different changes from the same SPEC, revise it.
