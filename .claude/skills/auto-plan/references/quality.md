# Plan Quality

Load this reference only before finalizing or refreshing `PLAN.md`.

## PLAN Anti-Patterns

- Vague slices: tasks like "improve robustness" or "clean up docs" without a produced artifact.
- Hidden dependencies: a slice needs context, files, or decisions not named in the plan.
- Untestable verification: checks that do not name commands or observable outcomes.
- Architecture theater: new structure introduced only to make the plan look sophisticated.
- Overloaded slices: one slice touching unrelated systems or too much to verify independently.

## Better Shape

- Give every material slice a concrete output and verification command.
- Make dependencies explicit and keep later slices out of the active slice.
- Prefer direct execution unless risk or file ownership justifies subagents.
- Split any slice that cannot be verified independently.

## Prose Hygiene

Plans attract architecture theater and signposting. Slice objectives should name what gets built, not announce what will be explored.

Scan for:
- "Let's explore", "we'll dive into": start with the verb
- "robust", "comprehensive", "elegant": name the specific property
- Vague attribution in architecture rationale ("best practice", "industry standard")
- -ing padding ("ensuring consistency", "enabling scalability")
- Generic positive framing of the approach

Before: "This slice explores implementing a robust, comprehensive caching layer — ensuring consistency, enabling scalability, and enhancing performance."
After: "Add a Redis read-through cache to the /users endpoint. Cache invalidates on write. TTL 5 minutes."

## Final Check

If the implementer must decide what "done" means, revise the slice.
