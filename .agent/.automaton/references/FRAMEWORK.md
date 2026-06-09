# Automaton Framework

Operating model for the stage-gated harness. Read once per session; skills assume this context.

## Stages

Five lifecycle stages: `frame → plan → execute → verify → verified`. `resume` re-enters at any point from durable state.

| Stage | Purpose | Produces |
|---|---|---|
| `frame` | Bound and de-risk the objective | `SPEC.md` |
| `plan` | Turn the spec into ordered, verifiable slices | `PLAN.md`, optional `DESIGN.md` |
| `execute` | Implement approved slices | Code, tests, docs, slice evidence |
| `verify` | Independent audit against acceptance criteria | Verification report, `VERIFY-GAP` on fail |
| `verified` | Terminal. Change is complete. | Completion summary |

## Skill Structure

Every skill follows this skeleton:

```
Preamble        -> identity, "does not" boundary, loading discipline
Quality Gate    -> checks before finalizing. Each skill ships a quality.md reference
Do              -> skill-specific procedure
Output          -> artifacts produced, state changes, handoff
Rules           -> guardrails
```

Conditional reference reads (`Read references/X.md when Y`) appear inline at their procedural trigger points.

## State Contract

- Machine state lives in `.agent/.automaton/state/current.json`.
- **Update state only through `sync-status.mjs`.** Never edit `current.json` by hand.
- Canonical pointers (`canonical_spec`, `canonical_plan`, `canonical_design`) and review verdicts are fields in `current.json`.
- Work artifacts live under `.agent/work/<change>/`; steering under `.agent/steering/`.

## Quality Gate

Every skill ships `references/quality.md` with three sections: anti-patterns, better shape, and prose hygiene. Read it when the skill's output drifts toward vagueness, theater, or inflation. The Quality Gate in each SKILL.md names the skill-specific trigger. All artifacts must pass `.agent/.automaton/references/ANTI-SLOP.md`.

## Hard Stop Tags

Two tags mark hard stops in skill procedures. Scan for them before reading the full `## Do` section.

- **`<GATE>`** -> prerequisite block. Do NOT proceed past this point unless all listed conditions are met. Used before an artifact write or a state mutation.
- **`<STOP>`** -> runtime halt. Halt immediately and report when any listed condition is true. Used when continuation would produce incorrect or unsafe output.

## Handoff Model

Two moves at every lifecycle edge:

- **Continue inline:** load and follow the next stage's contract in the same session. Default when the exit gate passes, reviews are non-blocking, and context is healthy.
- **Stop and hand off:** end the turn with a recommendation. Required at three edges: entry into `execute` (code changes need human authorization), entry into an optional review (`auto-ceo-review`, `auto-eng-review`), and verify outcomes (pass closes, fail returns to execute).

**Form.** Continue-inline emits no handoff line. The next contract's output speaks for it. A stop ends the turn with one line: `**Next:** <skill>, <reason in ≤8 words>`. Terminal completion reports `Change status: complete` and a `New objective:` line, with no `Next:`. The reason names the trigger, not the rule. Each mandatory stop's *why* is fixed above, so skills do not restate it.

## Loading Discipline

- Context is finite. Load progressively: smallest artifact first, more only when needed.
- Once a file is read in a session, do not re-read it unless it changed or verification requires fresh evidence.
- Artifacts (`SPEC.md`, `PLAN.md`) are reloadable contracts, not dossiers. Link detail under `spec/` or `slices/` instead of inlining everything.
