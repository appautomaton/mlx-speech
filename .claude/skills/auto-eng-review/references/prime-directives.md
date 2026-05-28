# Engineering Prime Directives

Non-negotiable standards for every engineering review.

1. **Zero silent failures.** Every failure mode must be visible: to the system, to the team, to the user. Silent failures are critical defects.

2. **Every error has a name.** Do not say "handle errors." Name the specific exception class, what triggers it, what catches it, what the user sees, and whether it is tested. Catch-all error handling is a code smell.

3. **Data flows have shadow paths.** Every data flow has a happy path and three shadow paths: nil input, empty/zero-length input, and upstream error. Trace all four.

4. **Interactions have edge cases.** Every user-visible interaction has edge cases: double-click, navigate-away-mid-action, slow connection, stale state, back button. Map them.

5. **Observability is proportional scope.** New codepaths need a way to diagnose failure. Dashboards, alerts, and runbooks are first-class only when the plan changes operated production behavior.

6. **Diagrams earn their space.** Add an ASCII diagram only when it clarifies a non-trivial flow, state transition, dependency graph, or rollout path that would be ambiguous in prose.

7. **Deferred work needs an owner surface.** Record deferred work only in the approved plan, roadmap, or review action when someone downstream will act on it. Do not create TODO files by default.

8. **Optimize for the 6-month future, not just today.** If this solves today's problem but creates next quarter's nightmare, say so.

9. **You have permission to say "scrap it and do this instead."** If there is a fundamentally better approach, table it now.

## Engineering Preferences

- DRY is important; flag repetition aggressively.
- Well-tested code is non-negotiable.
- Code should be "engineered enough," not under-engineered (fragile) and not over-engineered (premature abstraction).
- Err on the side of handling more edge cases, not fewer.
- Bias toward explicit over clever.
- Right-sized diff: smallest diff that cleanly expresses the change, but do not compress a necessary rewrite into a minimal patch.
- Observability is not optional, but the artifact should name the minimal signal needed for the change.
- Security is not optional when new inputs, privileges, secrets, network calls, or data classes are touched.
- Deployments are not atomic. Plan for partial states, rollbacks, and feature flags.
- ASCII diagrams in code comments only for complex designs where the diagram will stay maintained.
- Diagram maintenance is part of the change. Stale diagrams are worse than none.
