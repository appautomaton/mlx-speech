# Context Loading Discipline

Internal guidelines for preserving reasoning headroom across multi-session agentic work.

## Artifact Language Boundary

Loading is a decision you make during the session. It is not a fact the artifacts record.

**Keep artifacts concrete.** Do not write context-budget fields, token-allocation notes, or percentage estimates into SPEC.md, PLAN.md, slice detail files, or evidence blocks. Artifacts record objectives, acceptance criteria, verification, dependencies, status, evidence, risks, and links. Context-size estimates in PLAN.md are the common form of this mistake: when slice instructions outgrow the plan index, the answer is a `Detail: slices/slice-NNN.md` link, not a note about how large the slice is.

Report findings as conclusions rather than transcripts. The evidence you read stays in the session. The artifact carries what it proved.

## Progressive Loading Order

When entering any stage, load files in this order. Stop as soon as you have enough context to proceed.

```
1. .agent/.automaton/state/current.json   (always, tiny)
2. SPEC.md      (if canonical_spec exists)
3. PLAN.md      (if executing or verifying)
4. Linked detail files (spec/*.md and similar, only when referenced by spec or plan)
5. Source files (read those the current decision requires)
```

## Degradation Signals

You cannot reliably measure your own context usage. Watch behavior, not percentages:

- **Silent partial completion.** Work is claimed done but the implementation is incomplete.
- **Increasing vagueness.** "Appropriate handling" or "standard patterns" replace specific code and paths.
- **Skipped steps.** Protocol steps that would normally run are omitted.
- **Lost conclusions.** Re-deriving or contradicting something settled earlier in the session.

When the host surfaces actual context usage, treat it as corroboration: above roughly half, conserve. Near exhaustion, checkpoint. Do not guess percentages the host does not report.

## Conserve Then Checkpoint

Two responses, in order:

1. **Conserve.** Stop new wide reads. In skills that carry `references/HOST-TOOLS.md`, dispatch the librarian for lookups instead of reading inline. Summarize aggressively. Finish the current slice before starting anything new.
2. **Checkpoint.** When signals persist after conserving, record slice evidence and durable state, then stop with a clear next action. A clean checkpoint beats a degraded continuation.

## Re-Read Rule

Default: a file read this session stays usable from memory. Re-read it when any of these hold:

- The user asks you to.
- You wrote to it and need to verify the write.
- It is known to have changed.
- The current skill is an explicit verification pass and fresh evidence is part of the acceptance criteria.
- The session was compacted, or you are no longer sure what it said.

**If you cannot remember what a file said, re-read the specific section.** Answering from a confident guess is worse than the second read.

