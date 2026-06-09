---
name: auto-plan
description: Turn an approved spec into ordered slices. Use when framing is accepted and planning begins.
metadata:
  stage: plan
---

# auto-plan

Planning controller. Turns approved framing into ordered slices with verification commands.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-plan builds the smallest plan that makes execution safe while preserving the approved scope. It does not write code or broaden scope beyond the approved spec.

Loading discipline: hold SPEC.md, review state, and source files needed for accurate slices. Read wider project files when understanding existing code informs slice boundaries or verification commands. Read `.agent/.automaton/references/CONTEXT-BUDGET.md` when wider reads threaten context pressure. When locating code or tracing a flow would otherwise pull wide reads into context, you may dispatch the read-only `automaton-librarian` for a one-shot lookup (see `.agent/.automaton/references/LIBRARIAN.md`); it returns evidence, you keep the decision.

Artifact discipline: `PLAN.md` is the reloadable execution index, not the whole implementation dossier. Keep PLAN.md compact enough to re-read. For large coherent work, summarize slices in PLAN.md and link optional detail files under `.agent/work/<change>/slices/`. Split only for independent outcomes, not because one coherent plan has many requirements.

## Quality Gate

Before finalizing `PLAN.md`:
- Give every material slice a concrete output.
- Attach a verification command to every material slice.
- Name the execution topology: default continuation path, explicit checkpoints, subagent routes, and any parallel-safe groups.
- Remove vague tasks that do not define done.
- Read `references/quality.md` when the plan leaves execution decisions to the implementer.

## Do

### Context Loading

Load the canonical SPEC.md, linked spec detail that carries normative requirements, relevant DESIGN.md, and source files needed to choose slice boundaries, dependencies, and verification commands. Do not ignore linked `spec/*.md` files when they contain requirement IDs, gap IDs, invariants, audit questions, migration checkpoints, coverage targets, or acceptance detail.

### Assess Review State (if reviews exist)

If `product_review` exists in `current.json`, read `## Review: Product` in SPEC.md. Address each `approved_with_risks` risk in the plan. Stop and recommend `auto-frame` for `descoped` or `needs_clarification`.

If the engineering approach is complex or risky, recommend `auto-eng-review` before execution.

If SPEC.md contains content fields or produces writing, articles, briefs, decks, newsletters, documentation, or proposals, read `references/content-planning.md`; carry forward channel, source policy, factual risk, and format where they affect execution or verification.

If SPEC.md names requirement IDs, gap IDs, invariants, audit questions, migration checkpoints, or coverage targets, preserve them in PLAN.md and attach them to satisfying slices. Do not collapse traceable requirements into untraceable prose.

### Design Slices

Break work into ordered execution units, not topic buckets. Each slice must be:
- Testable: it produces an outcome that can be verified.
- Bounded: it can be executed and verified without loading unrelated slices.
- Independent: it can be executed without loading slices that come after it.
- Checkpointed only for human input: it marks a pause only when a human must act or choose before the next approved slice can start.

Read `references/slice-examples.md` when uncertain whether a slice is well-designed.

For content slices, also name the artifact target, allowed sources, factual-risk gate, and format constraint so `auto-execute` does not invent missing context.

Before writing slices, choose the execution topology: serial order, subagent routes, checkpoints, and parallel-safe groups. Continuation is the default after a verified slice; mark a checkpoint only when the agent must pause for human verification, a human decision, or a human action. Parallel-safe means dependencies are independent and write sets are disjoint; default to none. For multi-slice plans, make clear that execution should continue through all approved slices; execution windows are context-management batches, not planned stopping points.

Frame each slice with required fields first, then only the overrides the slice needs:

```
### Slice N: [Name]

Required:
**Objective:** [one sentence]
**Acceptance criteria:**
- [observable criterion]
**Verification:** [command or check that proves the slice is done]

Defaults, state only when overriding:
**Execution:** direct | subagent recommended | subagent required (default: direct)
**Depends on:** none
**Checkpoint after:** none | human-verify | decision | human-action (default: none)
**Checkpoint reason:** none

Include when useful:
**Touches:** [files, directories, or subsystems]
**Produces:** [specific artifact or state change]
**Detail:** [linked `slices/slice-NNN.md` file]
```

Rules:
- Every material slice must have a verification command. Verify the exact behavior, not the absence of errors. Include rollback verification for migrations.
- Every material slice must have acceptance criteria; execution cannot verify vibes.
- Omitted `Execution` means `direct`. Use `subagent recommended` for broad, cross-subsystem, interface, schema, or review-risk work. Use `subagent required` only for user-requested multi-agent execution or security-critical, production-data, or irreversible-state changes.
- Omitted `Depends on` means `none`.
- Continuation is the default. Omitted `Checkpoint after` means `none`, so the next slice may start after verification passes.
- Verification findings, implementation caveats, downstream consequences, and next-slice recommendations are not checkpoints when the approved plan already names the next slice. Record them as slice evidence or risks and continue.
- Checkpoint types (`human-verify`, `decision`, `human-action`) are defined once in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Checkpoint Semantics). Assign a checkpoint only when its definition holds; default to `none`.
- Keep slices small enough for one session. Move extended instructions to `slices/slice-NNN.md`; split only for independent outcomes.

### Write PLAN.md

Write the plan to `.agent/work/<change>/PLAN.md`.

**Core** sections (always present):
- **Goal**: one-line bounded goal or SPEC.md pointer; do not mirror the full SPEC text.
- **Ordered slice sequence**: dependency order, with linked detail files when needed.
- **Execution routing and topology**: default continuation path, explicit overrides/checkpoints, and a **Parallel-safe groups:** line set to `none` or the slice groups.
- **Per-slice verification**: one verification command inline on every material slice.

**Conditional** sections appear only when their trigger applies; omit or mark "n/a" otherwise:
- **Architecture approach:** introduces a new pattern, non-obvious decision, or cross-system integration. Omit when the design is obvious from SPEC.
- **Requirement traceability:** SPEC names gap IDs, invariant IDs, audit questions, migration checkpoints, or coverage targets. Omit when the SPEC has no traceable IDs.
- **Aggregate verification commands table:** ≥ 3 slices or commands not captured per-slice. Per-slice inline suffices for smaller plans (index over transcript).

Apply the Artifact Signal Discipline rules from `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` while writing: no mirror sections, index over transcript, append-replace not stack. Replace prior `## Review:` sections on re-run for the same change. Do not stack reviews.

### Write DESIGN.md (if non-trivial)

Write `.agent/work/<change>/DESIGN.md` only for non-trivial architecture or new patterns. Keep it under 200 lines; skip it when the approach is obvious from SPEC.

<GATE>

Do NOT write PLAN.md if:
- SPEC.md is missing or unreadable.
- `product_review` is `descoped` or `needs_clarification`.
- The scope is still ambiguous after reading SPEC.md.

If any of these are true, recommend `auto-frame` and stop.
</GATE>

### Update State

Run `node .agent/.automaton/scripts/sync-status.mjs --canonical-plan ".agent/work/<change>/PLAN.md" --stage plan` from the project root. Add `--canonical-design ".agent/work/<change>/DESIGN.md"` when DESIGN.md was written.

## Output

- `PLAN.md`: written to `.agent/work/<change>/PLAN.md`
- `DESIGN.md`: written to `.agent/work/<change>/DESIGN.md` (if needed)
- `.agent/.automaton/state/current.json`: records `canonical_design` (when written), `canonical_plan`, and `stage: plan` through `sync-status.mjs`
- Handoff (always stops): `Next: auto-eng-review` (optional review) or `Next: auto-execute`.

## Rules

- Prefer the smallest correct design.
- Remove placeholders instead of preserving them.
- Do not broaden scope to cover hypothetical future work.
- Preserve review sections on refresh unless the user explicitly requests consolidation.
- Every material slice must have acceptance criteria and an explicit verification command.
