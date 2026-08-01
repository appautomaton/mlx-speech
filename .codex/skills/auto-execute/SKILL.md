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

auto-execute owns execute-stage orchestration, route selection, state, and scope. Direct implementation and subagent implementation are two routes inside this skill. It does not reopen product scope or modify the approved plan's intent. Execute and verify one approved slice at a time inside the selected execution window. Plan-declared parallel-safe groups are the one exception to that serial order. Continuation is the default after a verified slice; checkpoints and STOP conditions are the exceptions. An execution window is a context-management batch, not a completion boundary.

Loading discipline: keep the active slice, execution-window metadata, acceptance criteria, route metadata, verification commands, and active files in context. Load linked detail files and traceability IDs for the active slice only; read wider project files only when implementation correctness requires it. Read `.agent/.automaton/references/CONTEXT-BUDGET.md` when wider reads threaten context pressure. When a lookup would otherwise pull wide reads into context, dispatch the read-only `automaton-librarian` (see `.agent/.automaton/references/LIBRARIAN.md`): it returns evidence, you keep the decision.

## Quality Gate

Before marking a slice complete:
- Keep edits inside the active slice.
- Investigate root cause before fixing bugs. Read `references/debug-protocol.md` only when bounded diagnosis needs more structure.
- Record verification evidence before advancing or selecting the next slice.
- Read `references/quality.md` when the diff looks clever, defensive, or broader than the plan requires.

## Do

<GATE>

Do NOT write code unless:
- `PLAN.md` is approved and `canonical_plan` in `.agent/.automaton/state/current.json` is set.
- `canonical_spec` still resolves: the spec chain holds end to end so cold resume can always load spec first.
- The next executable slice has an objective, acceptance criteria, and verification command.
- `engineering_review` is not `needs_correction` (otherwise stop and return to `auto-plan`).
- The route is direct, or the subagent route has passed its host capability check.
</GATE>

### Load State

Read the canonical `PLAN.md`. If it contains `VERIFY-GAP` annotations, treat those gap-fix objectives as the current work before selecting the next uncompleted slice.

If `engineering_review` is `approved_with_risks`, surface each risk's rationale before the slice it affects. The verdict already means safe to proceed: a named risk does not block its slice.

If the current slice involves prose, read `references/content-execution.md`. If it links `slices/slice-NNN.md` or requirement IDs in `spec/*.md`, load those linked files for the active slice and preserve their traceability IDs.

### Mark Execute Stage

After the canonical `PLAN.md` resolves and before changing code or project artifacts, run `node .agent/.automaton/scripts/sync-status.mjs --stage execute` from the project root. This records that the active change has entered execution while preserving the existing `canonical_plan`.

### Git Rhythm

Commit per verified slice when the working directory is a git repo. The verification gate is the authorization. Do not pause to ask. Read `references/git-rhythm.md` once at execute entry for detection, pre-existing dirt, and commit-failure handling, then run its entry check.

After slice verification passes in `Verify And Advance`, run `git add -A` followed by one of:

- `git commit -m "slice N: <objective>"` for a fresh slice (objective from `PLAN.md`).
- `git commit -m "slice N gap-fix: <fix objective>"` for a slice re-entered after `auto-verify` FAIL (fix objective from the `VERIFY-GAP` block).

**Strictly additive.** `git commit` only. Never `amend`, `reset`, `rebase`, `branch`, `checkout`, or `push`. One carve-out: coordinator-managed `git worktree add`/`remove` for parallel slice isolation, defined in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Git Rhythm). Subagents on the implementer route never run any git write command. The orchestrator owns history.

### Select Execution Window

The next slice is selected from `PLAN.md`. Build the smallest safe execution window:
- Always include the next uncompleted slice.
- Add following slices only while `Checkpoint after: none` is present or defaulted, dependencies are met, verification is explicit, and no STOP condition or context pressure appears.
- Execute the window serially by default. Cross-slice parallel dispatch is allowed only when `PLAN.md`'s **Parallel-safe groups:** line names the slices and write sets are disjoint, and in a git repo it runs under worktree isolation (`.agent/.automaton/references/SUBAGENT-PROTOCOL.md`, Parallel Isolation; mechanics in `references/git-rhythm.md`).

Omitted slice fields carry the defaults pinned in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Slice Defaults).

For each slice in the window, extract objective, dependencies, touched files or subsystems, constraints and anti-goals, acceptance criteria, verification commands, checkpoint metadata, route metadata, and linked detail files and traceability IDs. If a material slice is missing acceptance criteria or verification, stop and recommend `auto-plan`.

For content slices, also extract the content fields per `references/content-execution.md`. Stop only for the gaps it names: a missing required-core input, a missing source, or an unresolved factual-risk decision.

### Route Selection

The route decision lives here. The assignment criteria live in `auto-plan`. Honor the plan's `Execution:` value:
- `direct`: implement in the parent session.
- `subagent recommended`: prefer the subagent route.
- `subagent required`: use the subagent route. Do not implement directly.

Use the subagent route when the user explicitly requests multi-agent execution. If implementation reveals the assigned route no longer fits the slice, record a plan correction rather than silently rerouting. Do not make the user re-invoke execution for the same slice.

### Direct Route

Use this route only when route selection permits direct execution.

### Subagent Route

Use this route when `Execution` is `subagent required`, when `subagent recommended` is justified, or when the user requested multi-agent execution. Before the first dispatch, read `.agent/.automaton/references/SUBAGENT-PROTOCOL.md` and `references/HOST-TOOLS.md`: dispatch-by-name rules, role boundaries, status vocabulary, and host availability live there. Dispatch only the named host-native agents (`automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`) and fill the per-call slots from `references/implementer-prompt.md`, `references/spec-reviewer-prompt.md`, and `references/quality-reviewer-prompt.md`. The installed agent definitions carry the role bodies. Do not paste a role body into a generic worker or explorer agent.

If the host does not expose the named agents, fall back from `subagent recommended` to direct execution only when the slice remains safe. For `subagent required`, stop under the protocol's host-support condition and recommend `auto-plan` or a host change. Do not fall back to runtime-curated prompt injection.

Run the per-slice protocol as SUBAGENT-PROTOCOL.md defines it: dispatch from a packet built from the current slice only, verify evidence, then spec review before quality review, passing concrete reviewer issues back once through the `<requested-changes>` slot. Record a compact orchestration summary under `.agent/work/<change>/orchestration/` only when subagent or review details are needed later. The slice status still updates in place.

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
**Evidence:** changed `path`, command/result; key observation.
**Risks / next:** none, or one concrete item.
```

Append-replace the evidence block. Do not paste transcripts, full command logs, or source excerpts unless needed to explain a blocker.

After evidence is recorded, run the per-slice commit when the **Git Rhythm** is active. A failed commit is a STOP condition, not a step to skip.

Do not invent slice cursor or checkpoint fields in `.agent/.automaton/state/current.json`. Change state only through `node .agent/.automaton/scripts/sync-status.mjs` when stage, active change, review state, or canonical artifact pointers change.

If the completed slice has a checkpoint, validate it against the definitions (`human-verify`, `decision`, `human-action`) in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Checkpoint Semantics): it holds only when its defined condition is met. For checkpoint text that fails its definition, record a plan correction, keep the evidence, and continue when normal continuation conditions pass.

Continue within the selected execution window only when verification passed, dependencies are met, the next slice still matches the approved plan, context remains healthy, and no STOP condition applies. If the checkpoint is valid, pause with the next action and checkpoint reason.

### Continuation And Hand Off

When the selected execution window is complete but `PLAN.md` still has uncompleted approved slices, return to **Select Execution Window** immediately. "N slices remain" is progress state, not a stop reason. Remaining approved slices require another execution-window pass unless a valid checkpoint, STOP condition, context-pressure tier, or unavailable host capability prevents continuing.

If all slices are complete and no STOP condition applies, ensure slice evidence is recorded, then continue inline into `auto-verify`'s contract when safe. Do not make the user run `auto-verify` manually just because execution finished. Do not trust execute's own slice evidence as final verification. Continuing inline emits no handoff line: verify's own outcome speaks for both stages.

When execution cannot continue, the turn ends with a stop, never with silence. Report the slices completed this window, the concrete blocker, checkpoint, or STOP condition that halted it, and what the user must decide or do. Then end the turn with `**Next:** auto-execute, <reason in 8 words or fewer>` when approved slices remain, `**Next:** auto-verify, <reason>` when execution finished but continuation is unsafe, or `**Next:** auto-plan, <reason>` on a structural failure.

### Record Corrections

If implementation reveals a real mismatch between plan and reality, record the correction in `PLAN.md` on the current slice. Do not silently redefine the plan.

<STOP>

Halt immediately and report to the user when:
1. A dependency is missing and cannot be installed or resolved.
2. A test fails 3 times with the same error: identical failures mean the approach is wrong, not the run.
3. A plan instruction is ambiguous or contradictory and cannot be resolved with one clarifying question.
4. The approved slice no longer matches the codebase state.
5. The user asks for work outside the current slice.
6. Context degradation signals persist after conserving (response procedure: `.agent/.automaton/references/CONTEXT-BUDGET.md`, Conserve Then Checkpoint).
7. The plan requires subagents but the host cannot dispatch them.

Read `references/stop-examples.md` when uncertain whether a situation qualifies for STOP.
</STOP>

## Output

- Slice(s) executed: route used, files changed, commands run with results, and subagent statuses with review verdicts when the subagent route ran.
- Slice evidence updated in place: inline slice in `PLAN.md`, or linked detail file plus compact `PLAN.md` pointer.
- Per-slice commits when the Git Rhythm is active, in its pinned `slice N:` and `slice N gap-fix:` shapes.
- Execute stage recorded through `sync-status.mjs` when execution begins. No slice cursor field is added to current.json.

## Rules

- If the user asks for work outside the plan, reframe through `auto-frame` rather than bypassing it. The boundary is the change's evidence: slice evidence and per-slice commits have to describe what the plan approved, so an unplanned edit landing in the same diff makes the record wrong. The engagement criterion governs what starts a change, never what an active one absorbs.
