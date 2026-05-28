# Engineering Review Quality

Load this reference only before appending the engineering review to `PLAN.md`.

## Engineering Review Anti-Patterns

- Generic risk language: "may have edge cases" without naming the case.
- Missing evidence: concerns not tied to files, slices, commands, or architecture choices.
- Polite rubber-stamp: approval that ignores a weak test strategy or unclear data flow.
- Scope reopening: product redesign presented as engineering feedback.
- Unranked concerns: minor cleanup mixed with blockers.

## Better Shape

- Ground each concern in a plan slice, file area, command, or missing artifact.
- Distinguish blockers from follow-up suggestions.
- Keep product scope fixed unless an engineering issue makes the plan unbuildable.
- State what evidence would turn a risk into approval.

## Prose Hygiene

Engineering reviews attract generic risk language and vague concerns. Every finding should name a file, subsystem, or command.

Scan for:
- "could potentially cause issues": name the issue and where it happens
- "should be carefully considered": state the specific risk and its severity
- "robust error handling": name the exception, what catches it, what the user sees
- "comprehensive test coverage": name the test file and what it asserts
- Generic approval language ("the architecture is sound") without naming what was checked

Before: "The architecture is generally sound, though error handling should be carefully considered to ensure robust coverage of edge cases."
After: "Architecture fit: 8/10. Risk: parseToken() in src/auth.js catches all exceptions and returns null, so a malformed JWT is indistinguishable from an expired one. Add typed catches for JsonWebTokenError vs TokenExpiredError."

## Final Check

If the implementer cannot act on each finding, rewrite it or remove it.
