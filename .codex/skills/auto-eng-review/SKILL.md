---
name: auto-eng-review
description: Optional engineering go/no-go on a plan. Use when execution safety needs review before implementation.
metadata:
  stage: plan
---

# auto-eng-review

Optional engineering-safety review. Validates that a plan is safe to execute before implementation begins.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

Execution safety review. Architecture, data flow, edge cases, test strategy, not product vision. It does not change the plan's content or reopen product scope; its only write is appending its own `## Review: Engineering` section. Identifies risks that could cause failure, stalling, or rework.

A good review names the riskiest slice, the most likely failure mode, and whether the test strategy catches it. A bad review lists generic concerns.

Loading discipline: one PLAN.md read, optional DESIGN.md when `canonical_design` exists, one risk matrix, one verdict. Read source files when assessing technical risk: slice boundaries, dependency assumptions, and blast radius claims are only verifiable against the actual code.

## Quality Gate

Before appending the engineering review:
- Ground concerns in slices, file areas, commands, or missing artifacts.
- Separate blockers from follow-up cleanup.
- Avoid reopening product scope unless the plan is unbuildable.
- Read `references/quality.md` when findings are generic or unactionable.

## Do

<GATE>

Do NOT proceed unless:
- `canonical_plan` is set and `PLAN.md` is readable.

If the plan is missing or unreadable, set verdict to `needs_correction` and stop.
</GATE>

### Load State

Read the canonical `PLAN.md`. Read `DESIGN.md` only when `canonical_design` is set and resolves to a file. An unset pointer means the plan intentionally has no design artifact; continue without it. A set pointer with a missing file is stale: report it and continue (DESIGN.md is optional here).

### Restate the Plan

In engineering terms: what is being built, what systems does it touch, and what is the critical path?

### Evaluate Risks

Use this matrix as an internal checklist. Apply standards from `references/prime-directives.md` while evaluating.

### Risk Matrix

Architecture fit, data flow clarity, edge case coverage, test strategy, rollback safety, dependency risk.

A dimension where you can already name the concrete failure mode is a blocking concern, and naming it is the review's job. Surface it explicitly. Blocking concerns return `needs_correction`. Dimensions that clear the bar but carry named, slice-scoped risks return `approved_with_risks`. Read `references/engineering-sections.md` only when the plan carries non-trivial engineering risk.

### Render Verdict

Read `references/implementation-alternatives.md` only when PLAN.md lacks an approach rationale, the user asks for alternatives, or the verdict depends on comparing safer execution paths.

Use strict vocabulary: exactly one of the three approved values, no synonyms.

| Verdict | Meaning | Next Action |
|---------|---------|-------------|
| `approved` | Implementation is safe to proceed. | `auto-execute` |
| `approved_with_risks` | Implementation is safe but carries known risks. Document them. | `auto-execute` |
| `needs_correction` | Plan is flawed or unsafe. Return to planning. | `auto-plan` |

### Outside Voice

Optional. When a second model is reachable from this session and the plan carries non-trivial risk, read `references/outside-voice.md` after rendering the verdict: a round-capped cross-model loop that arbiters findings with logged reasons, surfaces tension to the user, and never auto-applies anything.

### Append Review

Add a `## Review: Engineering` section to `PLAN.md` using the exact template in `references/review-template.md`.

### Update State

Run `node .agent/.automaton/scripts/sync-status.mjs --engineering-review "<verdict>"` from the project root.

### Hand Off

The review always stops. The edge's why: `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Handoff Contract).

Report the verdict, the riskiest slice, and the one concern execution should watch for. Then end the turn with `**Next:** auto-execute, <reason>` for `approved` or `approved_with_risks`, or `**Next:** auto-plan, <reason>` for `needs_correction`.

## Output

- `PLAN.md` with appended `## Review: Engineering` section
- `.agent/.automaton/state/current.json` updated through `sync-status.mjs` with `engineering_review`; `stage` is unchanged by this skill

## Rules

- Do not emit the full risk matrix when all dimensions are acceptable. Keep the durable review to the review-template fields.
- Missing DESIGN.md is not a blocker when `canonical_design` is null, absent, or intentionally skipped by the plan.
