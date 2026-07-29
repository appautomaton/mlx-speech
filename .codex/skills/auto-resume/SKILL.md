---
name: auto-resume
description: Recover active change and next action from artifacts. Use on fresh session with existing work.
metadata:
  stage: resume
---

# auto-resume

Session recovery. Rebuilds context from durable artifacts, not memory or guessing.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-resume rebuilds context from durable artifacts, not from the user's description or the agent's training data. It does not modify artifacts, advance the stage, or start new work. It loads canonical artifacts in dependency order (spec first, then design, then plan) and reports what it found, what was blocked, and what comes next.

Loading discipline: start with artifacts needed for the current stage. Read project files when understanding the codebase helps rebuild accurate context for the next action. Read `.agent/.automaton/references/CONTEXT-BUDGET.md` when wider reads threaten context pressure.

## Quality Gate

Before producing the recovery summary:
- Trust durable artifacts over memory.
- Report stale pointers plainly.
- Recommend a next skill only when recovered state has incomplete or blocked work. For verified completion, report no next lifecycle skill.
- Read `references/quality.md` when the summary becomes narrative recap.

## Do

### Load State

<STOP>

Halt and report when:
- `.agent/` does not exist or `current.json` is missing.

Recommend `automaton install` to scaffold `.agent/`, then stop. Do not attempt recovery without a state file.
</STOP>

If work is complete or absent, read `.agent/steering/ROADMAP.md` only to surface pending phases as context.

### Verify Artifact Integrity

Check that `canonical_spec`, `canonical_design`, and `canonical_plan` resolve when present. If any pointer is stale, report it plainly. Recommend `auto-frame` for missing SPEC.md or `auto-plan` for missing PLAN.md.

### Load Artifacts

Treat `current.json` as the only source for active change, stage, and canonical artifact pointers. Load artifacts in dependency order and stop at the current stage. Read `references/artifact-order.md` for the full stage table.

### Reconcile Execution Ledger

When stage is `execute` or `verify` and the project is a git repo, read the execution ledger before summarizing: `git log --oneline -15` and `git status --porcelain`. Per-slice commits (`slice N: ...`, `slice N gap-fix: ...`) mark verified slices; match them against `PLAN.md` slice evidence. A dirty tree on top of the last slice commit is in-flight work for the next slice, not noise: name the touched files. When commits and `PLAN.md` evidence disagree, trust the commits and report the mismatch. Also run `git worktree list`: a stray worktree is the fingerprint of an interrupted parallel dispatch. Report it. Do not remove it.

### Surface Review State

If `current.json` contains `engineering_review`, read the `## Review: Engineering` section from the canonical plan and include it in the resume summary.

### Recovery Summary

Omit any line that would report nothing. A healthy resume is five lines, not ten fields of "none".

Always:

```
**Active change:** [name]
**Stage:** [frame|plan|execute|verify|verified]
**Artifacts loaded:** [list]
**What was done:** [1-2 sentences]
**What comes next:** [specific next action, or "change complete"]
```

Add only when it carries something:

```
**Blocked:** [what stopped, and on what]
**Execution ledger:** [last slice commit, in-flight files]
**Review verdicts:** [engineering: X]
**Missing state:** [stale pointer or absent artifact]
**Roadmap:** [N pending]
```

The goal is orientation, not transcription.

### Hand Off

Use `references/recovery-scenarios.md` for the full routing table. The invariant: recommend the next lifecycle skill only when recovered state is incomplete or blocked.

Resume orients and stops. It never starts the work it just found.

After the recovery summary, end the turn with `**Next:** <skill>, <reason>` when the recovered state has incomplete or blocked work. For a verified change, report `Change status: complete` and print no `Next:` line. When ROADMAP.md has pending items, surface them as optional future work rather than an automatic `auto-frame` handoff.

## Output

- Resume summary (the template above, nothing more)
- Artifacts loaded
- Review verdicts (if present)
- `.agent/.automaton/state/current.json` is read-only for auto-resume. Stale pointers are reported, not silently repaired
- Missing or conflicting state surfaces as a warning in the recovery summary.

## Rules

- Do not restart discovery when the current artifacts are sufficient.
