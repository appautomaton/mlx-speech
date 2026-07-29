# Execute Quality

Load this reference only before marking a slice complete or when editing code, tests, docs, or project artifacts.

**The test:** if the diff looks clever rather than inevitable from the plan, simplify it.

Failures that pass the test but still sink a slice:

- Obvious comments: prose that restates what the next line of code does.
- Defensive boilerplate: branches for impossible states, with no evidence from the codebase that they occur.
- Style drift: patterns that ignore local naming, error handling, or test conventions.
- Unrelated cleanup: opportunistic edits outside the active slice. Note them as follow-up instead.
- Evidence theater: claiming completion before verification exists.

Record what changed and what the verification observed, not how the work went.

Prose patterns: `.agent/.automaton/references/ANTI-SLOP.md`.
