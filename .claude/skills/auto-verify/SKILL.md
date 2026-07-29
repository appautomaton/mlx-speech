---
name: auto-verify
description: Verify completed plan against acceptance criteria. Use after all slices are executed.
metadata:
  stage: verify
---

# auto-verify

Verification gate. Independent audit of a completed plan. It runs once, not per-slice.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

Independent audit. Re-read the plan, run proof commands, and compare fresh results to acceptance criteria. It does not trust execute's self-assessment or fix what it finds. When continuing inline from execute, re-derive from fresh command output; execute's reasoning is context, not evidence.

Loading discipline: on warm continuation from execute, one PLAN.md read plus verification commands per criterion. On cold entry, load in dependency order, spec before plan; `auto-resume` owns the full order table. Read source files when verifying correctness requires inspecting the actual changes, not just command output.

## Quality Gate

Before writing the verification report:
- Tie every result to fresh command output or direct observation.
- Name skipped checks explicitly. Omission is not a pass.
- Treat partial evidence as FAIL for the plan.
- Read `references/quality.md` when the report sounds confident without proof.

## Do

### Load State

Read the canonical `PLAN.md`. Load only linked `slices/slice-NNN.md` files and referenced requirement IDs from `spec/*.md`: linked detail and traceability IDs are normative, and an unlinked supplemental file is not verification context. For prose slices, read `references/content-verification.md`.

### Mark Verify Stage

After `PLAN.md` resolves and before running commands, run `node .agent/.automaton/scripts/sync-status.mjs --stage verify` from the project root.

### Collect Acceptance Criteria

Gather every acceptance criterion and verification command from every slice in PLAN.md. Build a checklist: slice name → criterion → command. This is a plan-level audit.

<GATE>

Do NOT modify source code, tests, or project artifacts during verification. Verify reads and runs commands; its only writes are the markdown records this skill owns (`VERIFY-GAP` blocks, the `## Verification` section, the ROADMAP phase update). It does not fix.

Do NOT run any `git` write command (`commit`, `amend`, `reset`, `rebase`, `branch`, `checkout`, `worktree`, `push`). The commit rhythm is owned by `auto-execute` (see `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md`, Git Rhythm). Markdown writes that verify produces (`VERIFY-GAP` blocks on FAIL, the `## Verification` section and ROADMAP phase update on PASS) sit in the working tree; `auto-execute` sweeps them up on re-entry, or the user closes them after a terminal pass.
</GATE>

### Run Verification

Execute verification commands for each criterion. Mark each PASS, FAIL, or PARTIAL. If a criterion lacks a command, derive one from the acceptance criterion and document what you ran. For content slices, verify audience, thesis, voice, content anti-goals, channel, source policy, factual risk, format, and anti-slop scan with evidence.

### Evaluate

Binary: the plan passes only when every criterion across all slices passes. One FAIL means the plan fails.

### Report

Build the full criterion checklist internally. Use `references/verification-template.md` for report shape. Summarize passing criteria by slice. Expand failures, skipped checks, derived commands, PARTIAL results, or small 1-2 criterion plans.

### On Pass

- Append a compact `## Verification` section to the canonical `PLAN.md` (append-replace, never stack): per-slice criterion rollup, commands run, derived or skipped checks named, and the PASS verdict. Use the durable-record shape in `references/verification-template.md`. This is the record a future change or auditor reads. The inline report evaporates with the conversation.
- Run `node .agent/.automaton/scripts/sync-status.mjs --stage verified` from the project root. The verified sync disengages the harness: later session hooks stay quiet until a new objective starts.
- If `.agent/steering/ROADMAP.md` exists, mark the matching `change:` phase `status: done` per `.agent/.automaton/references/ROADMAP-CONTRACT.md`. Skip empty or non-matching phases. When no active or pending phase and no deferred item remains, reset to the contract's empty shape. The ROADMAP edit lands in the working tree as a markdown leftover. Do not commit it. The user closes it in their own rhythm.
- End the report with `Change status: complete` and a separate `New objective` line pointing to `auto-frame` for future work. Do not print a `Next:` line on PASS. Use `auto-resume` only for later re-entry or recovery.

### On Fail

Before annotating, check each failing criterion for an existing `VERIFY-GAP` block from a prior verification of this change. A repeat means the gap-fix cycle did not close it: the plan or spec is the suspect, not the implementation.

- First failure: annotate failed slices in `PLAN.md` with structured gap blocks, run `node .agent/.automaton/scripts/sync-status.mjs --stage execute` from the project root so re-entry resumes gap fixing, and hand off with `**Next:** auto-execute, <reason>`, which reads these annotations on re-entry.
- Repeated failure of the same criterion: annotate, run `node .agent/.automaton/scripts/sync-status.mjs --stage plan` from the project root, and hand off with `**Next:** auto-plan, <reason naming the repeated criterion>`.

Each gap block needs `VERIFY-GAP`, evidence, and a fix objective. Append-replace (`.agent/.automaton/references/FRAMEWORK.md`, Artifact Signal Discipline): replace prior `VERIFY-GAP` blocks for the same slice rather than stacking.

## Output

- Inline verification report; `PLAN.md` annotated with `VERIFY-GAP` blocks on failure, or closed with a durable `## Verification` section on pass
- State recorded in `current.json` through `sync-status.mjs`: `stage: verify` when verification starts, `stage: verified` on pass, `stage: execute` on fail, or `stage: plan` on a repeated fail of the same criterion
- `.agent/steering/ROADMAP.md` phase marked done on pass when applicable
- Warning-level findings surface to the verification report.

## Rules

- Verify what the plan requires. Flag an unmentioned common gap (input validation, concurrency, security) only when it is obviously critical to the change.
