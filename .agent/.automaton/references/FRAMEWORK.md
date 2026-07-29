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
Do              -> skill-specific procedure, ending in the stop-issuing step when the skill stops
Output          -> artifacts produced and state changes
Rules           -> guardrails
```

Conditional reference reads (`Read references/X.md when Y`) appear inline at their procedural trigger points.

## State Contract

- Machine state lives in `.agent/.automaton/state/current.json`.
- **Update state only through `sync-status.mjs`.** Never edit `current.json` by hand.
- Canonical pointers (`canonical_spec`, `canonical_plan`, `canonical_design`) and review verdicts are fields in `current.json`.
- Work artifacts live under `.agent/work/<change>/`; steering under `.agent/steering/`.
- Syncing a new `active_change` clears the prior change's canonical pointers and verdict. When current state shows a different unfinished change at `execute` or `verify`, name it and confirm parking it with the user before recording the new change.

## Quality Gate

Every skill ships `references/quality.md` with four sections: anti-patterns, better shape, prose hygiene, and a final check. Read it when the skill's output drifts toward vagueness, theater, or inflation. The Quality Gate in each SKILL.md names the skill-specific trigger. All artifacts must pass `.agent/.automaton/references/ANTI-SLOP.md`.

## GATE and STOP Tags

Two tags mark hard stops in skill procedures. Scan for them before reading the full `## Do` section.

- **`<GATE>`** -> prerequisite block. Do NOT proceed past this point unless all listed conditions are met. Used before an artifact write or a state mutation.
- **`<STOP>`** -> runtime halt. Halt immediately and report when any listed condition is true. Used when continuation would produce incorrect or unsafe output.

## Handoff Model

Two moves at every lifecycle edge:

- **Continue inline:** load and follow the next stage's contract in the same session. Default when the exit gate passes, reviews are non-blocking, and context is healthy.
- **Stop and hand off:** end the turn with a recommendation. Required at the four stop edges pinned in `ARTIFACT-LIFECYCLE.md` (Handoff Contract); a skill does not restate an edge's why.

**Form.** Continue-inline emits no handoff line. The next contract's output speaks for it. A stop ends the turn with one line: `**Next:** <skill>, <reason in ≤8 words>`. Terminal completion reports `Change status: complete` and a `New objective:` line, with no `Next:`. The reason names the trigger, not the rule.

**Where.** A skill that stops issues this line from a step inside `## Do`, after its artifact writes and state mutations, never from `## Output`. The conventional shape is a terminal `### Hand Off` step; a stop that fans out by outcome or loops back into the procedure issues from the step that owns the outcome instead. A handoff listed among artifacts reads as a manifest entry, not an instruction. Elsewhere, name a target skill in plain text and reserve the `**Next:**` form for the emitted line.

## Asking The User

Ask one question per message, with your recommended answer and its reason attached, so a single "yes" keeps the conversation moving. For a branch decision, offer 2 to 4 concrete options with a one-line reason each. Prefer the host question tool: its name, schema, and availability gate are on the `questions` line of `HOST-TOOLS.md` in your skill's `references/`. When it is not among your tools this turn, ask in plain text and end the turn.

## Loading Discipline

Context is finite: load progressively, recall over re-read (home: `.agent/.automaton/references/CONTEXT-BUDGET.md`). Artifacts are reloadable indexes, not dossiers; layout and linking rules live in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Progressive Disclosure).

## Artifact Signal Discipline

Automaton artifacts are read by future skills and humans. Every section must change a downstream decision.

1. **No mirror sections** -> one concept per section. If two sections answer the same question, delete one or reframe them.
2. **Index over transcript** -> aggregate tables (traceability, verification rollups, slice summaries) earn their place only at ≥ 3 entries. For 1–2 entries, inline the information where it is used.
3. **Core versus conditional sections** -> lifecycle SKILL.md required-section lists distinguish core (always present) from conditional (include only when the named trigger applies). Each conditional section names its trigger.
4. **Append-replace, not stack** -> the skill that owns a review section or gap block replaces its own prior block on re-run for the same change, not stacked. A producing skill that refreshes SPEC.md or PLAN.md preserves every existing `## Review:` section.
5. **Inline default for transient reports** -> status summaries and intermediate audit output live in the conversation only. Write to disk only when a future skill or human will read it again: the terminal `## Verification` section on PLAN.md (pass) and `VERIFY-GAP` blocks (fail) are the named exceptions because re-entry and audit consume them.

**Deletion test for any section:** if this section were removed, what downstream skill or human loses information? If nothing, drop it.
