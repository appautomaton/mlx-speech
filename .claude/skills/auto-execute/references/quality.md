# Execute Quality

Load this reference only before marking a slice complete or when editing code, tests, docs, or project artifacts.

## Execute Anti-Patterns

- Obvious comments: prose that restates what the next line of code does.
- Unnecessary abstraction: helpers, wrappers, or indirection not required by the plan.
- Defensive boilerplate: branches for impossible states without evidence from the codebase.
- Style drift: patterns that ignore local naming, error handling, or test conventions.
- Unrelated cleanup: opportunistic edits outside the active slice.
- Evidence theater: claiming completion before verification exists.

## Better Shape

- Match existing local patterns before introducing a new one.
- Keep edits within the touched files or subsystems named by the slice.
- Record concrete evidence: files changed, commands run, outcomes observed.
- If cleanup is needed but unrelated, note it as follow-up instead of doing it.

## Prose Hygiene

Execution artifacts attract obvious comments and inflated summaries. Orchestration notes should record what happened, not sell what was accomplished.

Scan for:
- Comments that restate what the next line does
- "successfully" in completion notes: either it passed or it failed
- "crucial fix", "important improvement": name the bug or delta
- Sycophantic openers in subagent dispatch ("Great work on...")
- -ing summaries ("highlighting the completion of...", "ensuring the stability of...")

Before: "Successfully implemented the crucial authentication middleware, ensuring robust security across all endpoints."
After: "Added verifyToken middleware to 4 protected routes. Tests pass. 401 on invalid token confirmed."

## Final Check

If the diff looks clever rather than inevitable from the plan, simplify it.
