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

Execution safety review. Architecture, data flow, edge cases, test strategy, not product vision. It does not modify the plan or reopen product scope. Identifies risks that could cause failure, stalling, or rework.

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

Read the canonical `PLAN.md`. Read `DESIGN.md` only when `canonical_design` is set and resolves to a file; otherwise continue without it and note that the plan intentionally has no design artifact.

### Restate the Plan

In engineering terms: what is being built, what systems does it touch, and what is the critical path?

### Evaluate Risks

Use this matrix as an internal checklist. In chat, summarize only the verdict-driving dimensions unless the user asks for the full matrix. Apply standards from `references/prime-directives.md` while evaluating.

### Risk Matrix

Read `references/risk-examples.md` for sample matrix scoring.

| Dimension | Rating (0–10) | What a 10 looks like |
|-----------|---------------|----------------------|
| Architecture fit | | Clean integration, no hacks, follows existing patterns |
| Data flow clarity | | Every input, transform, and output is traceable |
| Edge case coverage | | Failure modes are enumerated and handled |
| Test strategy | | Tests are specified before code, not after |
| Rollback safety | | Can revert without data loss or downtime |
| Dependency risk | | No new critical dependencies; existing ones are stable |

A score ≤ 3 in any dimension is a blocking concern. Surface it explicitly. Read `references/engineering-sections.md` only when the plan carries non-trivial engineering risk.

### Render Verdict

Use exactly one of the three approved values. Read `references/implementation-alternatives.md` only when PLAN.md lacks an approach rationale, the user asks for alternatives, or the verdict depends on comparing safer execution paths.

### Verdict Values

Use strict vocabulary. No synonyms.

| Verdict | Meaning | Next Action |
|---------|---------|-------------|
| `approved` | Implementation is safe to proceed. | `auto-execute` |
| `approved_with_risks` | Implementation is safe but carries known risks. Document them. | `auto-execute` |
| `needs_correction` | Plan is flawed or unsafe. Return to planning. | `auto-plan` |

### Append Review

Add a `## Review: Engineering` section to `PLAN.md` using the exact template in `references/review-template.md`.

### Update State

Run `node .agent/.automaton/scripts/sync-status.mjs --engineering-review "<verdict>"` from the project root.

### Recommend

State the next skill based on the verdict.

## Output

- `PLAN.md` with appended `## Review: Engineering` section
- `.agent/.automaton/state/current.json` updated through `sync-status.mjs` with `engineering_review`; `stage` is unchanged by this skill
- Handoff (verdict-mapped, always stops): `approved`/`approved_with_risks` → `Next: auto-execute`; `needs_correction` → `Next: auto-plan`.

## Rules

- Focus on execution safety, not product vision.
- Prefer specific engineering objections over generic caution.
- Do not broaden scope just to feel thorough.
- Do not emit the full risk matrix when all dimensions are acceptable; keep the durable review to the 5-field template.
- Verdict vocabulary is strict. Use only the three approved values.
- Missing DESIGN.md is not a blocker when `canonical_design` is null, absent, or intentionally skipped by the plan.
