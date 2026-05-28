---
name: auto-execute
description: Implement approved plan slices. Use as the execute-stage entry point.
metadata:
  stage: execute
---

# auto-execute

Implementation controller. Executes approved plan slices without reopening product scope.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-execute owns execute-stage orchestration, route selection, state, and scope. Direct implementation and subagent implementation are two routes inside this skill. It does not reopen product scope or modify the approved plan's intent. Execute and verify one approved slice at a time inside the selected execution window. Continuation is the default after a verified slice; checkpoints and STOP conditions are the exceptions. An execution window is a context-management batch, not a completion boundary.

Loading discipline: keep the active slice, execution-window metadata, acceptance criteria, route metadata, verification commands, and active files in context. Load linked detail files and traceability IDs for the active slice only; read wider project files only when implementation correctness requires it. Read `.agent/.automaton/references/CONTEXT-BUDGET.md` when wider reads threaten context pressure.

## Quality Gate

Before marking a slice complete:
- Keep edits inside the active slice.
- Investigate root cause before fixing bugs; read `references/debug-protocol.md` only when bounded diagnosis needs more structure.
- Record verification evidence before advancing or selecting the next slice.
- Read `references/quality.md` when the diff looks clever, defensive, or broader than the plan requires.

## Do

<GATE>

Do NOT write code unless:
- `PLAN.md` is approved and `canonical_plan` in `.agent/.automaton/state/current.json` is set.
- The next executable slice has an objective, acceptance criteria, and verification command.
- `engineering_review` is not `needs_correction` (otherwise stop and return to `auto-plan`).
- The route is direct, or the subagent route has passed its host capability check.
</GATE>

### Load State

Read the canonical `PLAN.md`. If it contains `VERIFY-GAP` annotations, treat those gap-fix objectives as the current work before selecting the next uncompleted slice.

If `engineering_review` is `approved_with_risks`, surface the rationale before starting but block only when the risk affects the current slice.

If the current slice involves prose, read `references/content-execution.md`. If it links `slices/slice-NNN.md` or requirement IDs in `spec/*.md`, load those linked files for the active slice and preserve their traceability IDs.

### Mark Execute Stage

After the canonical `PLAN.md` resolves and before changing code or project artifacts, run `node .agent/.automaton/scripts/sync-status.mjs --stage execute` from the project root. This records that the active change has entered execution while preserving the existing `canonical_plan`.

### Git Rhythm

Commit per verified slice when the working directory is a git repo. The verification gate is the authorization; do not pause to ask.

**Detect once at entry.** After `Mark Execute Stage` resolves, run `git rev-parse --git-dir` and `git status --porcelain`. The rhythm is inactive — silently, for the rest of the run — when:

- the directory is not a git repo;
- the user has told this run not to use git;
- the repo is mid-rebase, mid-merge, mid-cherry-pick, mid-bisect, or on detached HEAD.

**Pre-existing dirt.** If `git status` reports uncommitted changes at entry, announce once in the conversation that slice 1's commit will sweep them in, then proceed without asking. The rhythm matches what `git add -A && git commit` would do manually; recovery (`git reset HEAD~`) is in the user's normal toolkit.

**Commit per verified slice.** After slice verification passes in `Verify And Advance`, run `git add -A` followed by one of:

- `git commit -m "slice N: <objective>"` for a fresh slice (objective from `PLAN.md`).
- `git commit -m "slice N gap-fix: <fix objective>"` for a slice re-entered after `auto-verify` FAIL (fix objective from the `VERIFY-GAP` block).

**Strictly additive.** `git commit` only. Never `amend`, `reset`, `rebase`, `branch`, `checkout`, or `push`. Subagents on the implementer route never run any git command — the implementer prompt enforces this; the orchestrator owns history.

If the commit operation itself fails (pre-commit hook rejection, signing failure, repo entering an interrupted state mid-run), STOP and surface the failure verbatim. Do not retry with workarounds; do not silently skip the rhythm to keep going.

See `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Git Rhythm) for the cross-skill contract.

### Select Execution Window

The next slice is selected from `PLAN.md`. Build the smallest safe execution window:
- Always include the next uncompleted slice.
- Add following slices only while `Checkpoint after: none` is present or defaulted, dependencies are met, verification is explicit, and no STOP condition, slice-blocking review risk, or context pressure appears.
- Execute the window serially by default. Cross-slice parallel dispatch is allowed only when `PLAN.md`'s **Parallel-safe groups:** line names the slices and write sets are disjoint.

Slice defaults:
- Missing `Execution` means `direct`.
- Missing `Depends on` means `none`.
- Missing `Checkpoint after` means `none`.
- Missing checkpoint reason means `none`.

For each slice in the window, extract objective, dependencies, touched files or subsystems, constraints and anti-goals, acceptance criteria, verification commands, checkpoint metadata, route metadata, and linked detail files and traceability IDs. If a material slice is missing acceptance criteria or verification, stop and recommend `auto-plan`.

For content slices, also extract artifact target, audience, thesis, voice, content anti-goals, channel, source policy, factual risk, and format. If the slice needs a missing source or factual-risk decision, stop with `NEEDS_CONTEXT`.

### Route Selection

The route decision lives here:
- `direct`: small area, no slice-blocking review risk, fits in the parent session.
- `subagent recommended`: prefer subagents when the slice crosses subsystem boundaries, touches many files, modifies shared interfaces or data schemas, or carries a relevant `approved_with_risks` verdict.
- `subagent required`: use the subagent route. Do not implement directly.

Use the subagent route when the user explicitly requests multi-agent execution. Do not tell the user to invoke another execute skill for the same slice.

### Direct Route

Use this route only when route selection permits direct execution. Change code and project artifacts in the order the slice requires. Keep diffs small, local, and easy to verify. For prose artifacts, follow `references/content-execution.md`.

### Subagent Route

Use this route when `Execution` is `subagent required`, when `subagent recommended` is justified, or when the user requested multi-agent execution. Before dispatching, read `.agent/.automaton/references/SUBAGENT-PROTOCOL.md` and `references/HOST-TOOLS.md`. Dispatch only the named host-native agents listed in `HOST-TOOLS.md` — `automaton-implementer`, `automaton-spec-reviewer`, and `automaton-quality-reviewer` — and fill the per-call slots from `references/implementer-prompt.md`, `references/spec-reviewer-prompt.md`, and `references/quality-reviewer-prompt.md`. The static role bodies live in the host-native agent definitions; do not paste a role body into a generic worker or explorer agent.

If `HOST-TOOLS.md` says subagents are unavailable, fall back from `subagent recommended` to direct execution only when the slice remains safe. For `subagent required`, stop and recommend `auto-plan` or a host/configuration change. If a named agent is configured out of the host (Codex `[features].multi_agent` disabled, OpenCode `permission.task` denied for `automaton-*`, Claude agent file missing), treat the host as not exposing subagent support and stop — do not fall back to runtime-curated prompt injection into a generic agent.

Run the per-slice protocol:
1. Build a dispatch packet from the current slice only.
2. Dispatch the implementer.
3. Provide at most one targeted context correction for `NEEDS_CONTEXT`.
4. Verify expected file changes before spec review.
5. Run spec review before code-quality review.
6. Send concrete reviewer issues to an implementer once, then re-review.
7. Record a compact orchestration summary under `.agent/work/<change>/orchestration/` only when subagent/review details are needed later. The slice status still updates in place.

Do not mark the slice complete unless implementation status is acceptable, spec review is `APPROVED`, quality review is `APPROVED`, and slice verification evidence exists.

### Verify And Advance

Run the narrowest useful checks as soon as they can fail. Prefer targeted checks over full-suite rituals until the slice is stable.

Record completion evidence in place:
- If the slice is inline in `PLAN.md`, update that slice entry in `PLAN.md`.
- If the slice has `Detail: slices/slice-NNN.md`, update that linked detail file and keep a compact `PLAN.md` pointer.
- Do not create separate execution evidence files by default.

Use this compact evidence shape:

```markdown
**Status:** complete | blocked | needs-plan-correction
**Evidence:** changed `path`; command/result; key observation.
**Risks / next:** none, or one concrete item.
```

Append-replace the evidence block. Do not paste transcripts, full command logs, or source excerpts unless needed to explain a blocker.

After evidence is recorded, run the per-slice commit when the **Git Rhythm** is active. A failed commit is a STOP condition, not a step to skip.

The next slice is selected from `PLAN.md`; do not invent slice cursor or checkpoint fields in `.agent/.automaton/state/current.json`. Change state only through `node .agent/.automaton/scripts/sync-status.mjs` when stage, active change, review state, or canonical artifact pointers change.

If the completed slice has a checkpoint, validate that it actually requires human input per the checkpoint definitions (`human-verify`, `decision`, `human-action`) in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Checkpoint Semantics): a checkpoint holds only when its defined condition is met.

Do not pause for checkpoint text that only records verification findings, implementation caveats, downstream consequences, known limitations, or a recommendation for the next already-approved slice. Record a plan correction, keep the evidence, and continue when normal continuation conditions pass.

Continue within the selected execution window only when verification passed, dependencies are met, the next slice still matches the approved plan, context remains healthy, and no STOP condition applies. If the checkpoint is valid, pause with the next action and checkpoint reason.

### Continuation And Handoff

When the selected execution window is complete but `PLAN.md` still has uncompleted approved slices, return to **Select Execution Window** immediately. "N slices remain" is progress state, not a stop reason. Remaining approved slices require another execution-window pass unless a valid checkpoint, STOP condition, context-pressure tier, or unavailable host capability prevents continuing.

If all slices are complete and no STOP condition applies, ensure slice evidence is recorded, then continue inline into `auto-verify`'s contract when safe. Do not make the user run `auto-verify` manually just because execution finished. When continuing, re-read the canonical `PLAN.md`, collect every acceptance criterion, run or derive verification commands, and produce the verification report. Do not trust execute's own slice evidence as final verification.

### Record Corrections

If implementation reveals a real mismatch between plan and reality, record the correction in `PLAN.md` on the current slice. Do not silently redefine the plan.

<STOP>

Halt immediately and report to the user when:
1. A dependency is missing and cannot be installed or resolved.
2. A test fails repeatedly (> 3 attempts) with the same error.
3. A plan instruction is ambiguous or contradictory and cannot be resolved with one clarifying question.
4. The approved slice no longer matches the codebase state.
5. The user asks for work outside the current slice.
6. Context pressure reaches DEGRADING or EMERGENCY.
7. The plan requires subagents but the host cannot dispatch them.

Read `references/stop-examples.md` when uncertain whether a situation qualifies for STOP.
</STOP>

## Output

- Slice(s) executed and route used: direct, subagent recommended, or subagent required.
- Files changed with one-line rationale per file.
- Commands run and results.
- Subagent statuses and review verdicts when used.
- Slice evidence updated in place: inline slice in `PLAN.md`, or linked detail file plus compact `PLAN.md` pointer.
- Per-slice commits when the Git Rhythm is active: `slice N: <objective>` for fresh slices, `slice N gap-fix: <fix objective>` for re-entries after a verify FAIL.
- Execute stage recorded through `sync-status.mjs` when execution begins; no slice cursor field is added to current.json.
- Execution window checkpoint or stop reason when continuation pauses; if approved slices remain, name the valid blocker that prevents continuing.
- Verification report when all slices complete and continuation is safe; otherwise recommended next skill: `auto-execute` (slices remain), `auto-verify` (execution complete but continuation blocked), or `auto-plan` (structural failure).

## Rules

- auto-execute owns route selection and execution-window continuation.
- Build an execution window, but execute and verify one slice at a time.
- Serial execution is the default; parallel cross-slice dispatch requires explicit plan approval and disjoint write sets.
- Do not silently redefine the plan; record corrections transparently.
- auto-execute owns all `git commit` operations for Automaton; the rhythm is strictly additive (no `amend`, `reset`, `rebase`, `branch`, `checkout`, `push`), and subagents never run git.
- If the user asks for a quick fix outside the plan, reframe through `auto-frame`; do not bypass the plan.
- Keep durable evidence in `PLAN.md` or linked `slices/slice-NNN.md`, not new evidence files by default.
