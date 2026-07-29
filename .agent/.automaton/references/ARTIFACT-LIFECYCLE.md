# Artifact Lifecycle

Shared contract for what each Automaton stage consumes, writes, records, and hands off. This reference guides skills. It does not add runtime enforcement.

## Invariants

Stage list, state contract, and `sync-status.mjs` mandate are in `FRAMEWORK.md`.

- Concrete state paths live in `current.json`; artifact-to-artifact links live inside `SPEC.md` and `PLAN.md`. Do not create a separate status prose artifact to mirror them.
- Skills write artifacts only for the active change unless a skill explicitly documents a steering output.
- Do not add archive behavior or runtime lifecycle machinery here. The STOP Conditions list below names the full ban.

## Progressive Disclosure

`SPEC.md` and `PLAN.md` are canonical indexes, not forced compression targets. For large coherent work, keep canonical files reloadable and link normative detail files instead of narrowing the goal.

Allowed active-change layout:

```text
.agent/work/<change>/SPEC.md
.agent/work/<change>/spec/*.md
.agent/work/<change>/PLAN.md
.agent/work/<change>/slices/*.md
.agent/work/<change>/DESIGN.md
.agent/work/<change>/orchestration/*.md   # conditional: subagent route or complex review loops only (e.g. outside-voice-log.md)
```

- `auto-frame` records `canonical_spec` only once the spec is complete, so a `SPEC.md` without the pointer means framing is still in progress.
- `SPEC.md` must summarize and link every normative `spec/*.md` detail file. Unlinked supplemental files are notes, not contract.
- `PLAN.md` must link any `slices/*.md` detail file and preserve requirement IDs, gap IDs, invariants, audit questions, migration checkpoints, or coverage targets from SPEC.md.
- Execute and verify load only detail files linked to the active slice or requirement IDs.
- Execute writes slice evidence in place: inline slices update `PLAN.md`; linked detail slices update `slices/slice-NNN.md`; `orchestration/*.md` is supporting evidence, not the default write target.
- Split a change only for independent outcomes. Do not split or narrow one coherent outcome solely because the spec or plan has many files, gaps, constraints, or scenarios.
- If a skill narrows the user's stated scope, it must name the narrowing, explain why, and then widen the scope, ask the user to confirm, or record the deferred scope as a `Deferred / Not in scope` note inside the current change's SPEC. A narrowed scope never becomes a `ROADMAP.md` phase; phase authorship rules live in `.agent/.automaton/references/ROADMAP-CONTRACT.md`.

## Stage Handoffs

| Stage | Required inputs | Produces | State pointer expectations | Next handoff |
| --- | --- | --- | --- | --- |
| `frame` | the request, conversation, and repo evidence | `SPEC.md`, plus a `ROADMAP.md` update when the user approves a phased decomposition | `auto-frame` sets `active_change`, `stage: frame`, and `canonical_spec`; `stage` stays `frame`, and auto-plan records `stage: plan` when it writes PLAN.md | **Stop** → `auto-plan` once the user approves SPEC.md. |
| `plan` | `canonical_spec`; optional review sections | `.agent/work/<change>/PLAN.md`; optional `DESIGN.md` | `canonical_plan` points to PLAN.md; `canonical_design` only when DESIGN.md exists; `stage` becomes `plan` | **Stop** → `auto-eng-review` (optional) or `auto-execute`. |
| `execute` | approved PLAN.md (with `canonical_spec` still resolving), current slice, acceptance criteria, verification commands | code/docs/tests plus PLAN-required slice evidence | auto-execute sets `stage: execute` after `canonical_plan` resolves and before changes; do not change canonical pointers to missing files; do not add slice cursor state | **Continue** → re-enter for remaining slices, then `auto-verify` when all complete; **stop** at a valid checkpoint, STOP condition, context pressure, or host limit. |
| `verify` | canonical PLAN.md (with `canonical_spec` still resolving), executed slices, verification commands | verification report; `VERIFY-GAP` annotations in PLAN.md on failure; terminal `## Verification` section in PLAN.md on pass | auto-verify sets `stage: verify` after `canonical_plan` resolves and before commands; failure returns state to `stage: execute`, or to `stage: plan` when the same criterion fails a second consecutive verification | **Stop** → `verified` on pass (terminal); `auto-execute` on fail; `auto-plan` on a repeated fail of the same criterion. |
| `verified` | canonical PLAN.md and verification evidence | completed change summary; roadmap phase marked done when applicable | `stage: verified` set only on full verification pass | None. Terminal. `auto-frame` only as a new-objective entry point. |
| `resume` | current state and canonical artifact pointers | concise recovery summary and next recommended skill | does not invent missing pointers; stale pointers are reported, not silently repaired | Orient and stop → the skill matching recovered state. |

## Handoff Contract

The two-move model (**Continue inline** / **Stop and hand off**) is in `FRAMEWORK.md`. Continue inline by default so a clean handoff does not force the user to re-invoke the next skill. This is not nested skill invocation (DD-003): no skill calls another. The agent loads the next stage's SKILL.md and proceeds. Do not invent a universal Skill tool or hidden dispatcher.

**Stop and hand off at four edges:**

1. **Frame's exit** -> the user reads and approves SPEC.md before planning begins. The human reading the spec is the product review. Covers `frame → plan`.
2. **Entry into `execute`** -> code and project artifacts start changing there, so a human authorizes it. Covers `plan → execute`, `auto-eng-review → execute`, and a failed `verify → execute`.
3. **Entry into the optional `auto-eng-review`** -> user-invoked. A producing skill recommends the review and stops. It does not auto-run a review on the artifact it just wrote, which would trap the review in the producer's own context.
4. **Verify outcomes** -> a pass closes the change, a fail returns to execute, a repeated fail of the same criterion returns to plan. Stop in every case.

`auto-verify` is the mandatory gate, not an optional review, so `execute → verify` continues inline. The audit re-derives from fresh command output, never from execute's reasoning. `auto-resume` orients and stops: it reports findings and recommends a next skill rather than continuing, so the user keeps the direction. `stage: verified` is terminal. Any `auto-frame` mention at `verified` is for a new objective, not a same-change handoff.

Each handoff carries five durable elements:

1. **Exit gate** -> condition required to advance.
2. **Artifacts produced or updated** -> files written for the active change.
3. **State mutation** -> `current.json` fields changed through `sync-status.mjs`: `stage`, canonical pointers, or review verdicts.
4. **Diagnostic handling** -> `error` diagnostics block advancement. `warning` diagnostics surface to the next stage.
5. **Next-stage recommendation, blocker, or completion note** -> what to invoke next, what blocks progress, or that the active change is complete.

## Checkpoint Semantics

`Checkpoint after:` marks a slice that must pause for human input before the next slice starts. The label vocabulary is pinned in `contracts-data.json`. Each value's meaning is defined here, once, so `auto-plan` (which assigns checkpoints) and `auto-execute` (which honors them) cannot drift.

- **`none`** (default) -> no pause. The next slice may start once verification passes.
- **`human-verify`** -> valid only when available commands, tests, host tools, and local inspection cannot verify the result. If any of those can confirm it, it is not a checkpoint.
- **`decision`** -> valid only when a human must choose among named product, architecture, design, scope, or risk options before the next slice can start, and the answer changes that next slice. The checkpoint reason must state the concrete question and the options. Not for reversible engineering judgment, known limitations, verification findings, or "the next slice should be…" notes.
- **`human-action`** -> valid only when progress requires an external action the agent cannot perform, such as 2FA, account approval, or off-machine access.

Verification findings, implementation caveats, downstream consequences, and recommendations for an already-approved next slice are not checkpoints. Record them as slice evidence or risks and continue.

## Slice Defaults

Omitted `PLAN.md` slice fields default as pinned here, so plan and execute cannot drift:

- Omitted `Execution` means `direct`.
- Omitted `Depends on` means `none`.
- Omitted `Checkpoint after` means `none`.
- Omitted checkpoint reason means `none`.

## Git Rhythm

Per-slice commits are owned by `auto-execute`, whose SKILL.md holds the operational rhythm. This contract pins the cross-skill invariants:

- **Single owner.** `auto-execute` runs every `git commit` Automaton produces. `auto-verify` never invokes any git write command. Its read-only-on-code gate extends to git history. Subagents on the implementer route never invoke any git write command. Reads like `git log` stay available as context. The orchestrator owns history.
- **Strictly additive.** `git commit` only. Never `amend`, `reset`, `rebase`, `branch`, `checkout`, or `push`. The harness never rewrites history a user might already have inspected. One carve-out: coordinator-managed `git worktree add`/`remove` for parallel slice isolation does not breach this rule, because the user's checked-out branch is never switched and every result lands as a normal additive slice commit on it.
- **auto-verify leftovers.** Markdown writes from `auto-verify` are not committed by the producing skill. `VERIFY-GAP` blocks added to PLAN.md on FAIL fold into the next `auto-execute` gap-fix commit on re-entry. The PASS leftovers (the `## Verification` section on PLAN.md and the one-line ROADMAP phase update) sit in the working tree as terminal-state notes. The user closes them in their own cadence.
- **Recovery ledger.** The per-slice commit trail is the durable execution cursor. `auto-resume` reads it (read-only) on cold re-entry to reconcile `PLAN.md` slice evidence with reality: the last `slice N:` commit marks the last verified slice, and a dirty tree on top of it is in-flight work for the next slice.

Validation tier: L3 (prompt prose plus regression tests). No runtime enforcement, so the rhythm stays portable across hosts.

## Review Verdict Routing

`auto-eng-review` is an optional lifecycle check, not a stage prerequisite. Downstream skills must respect any review verdict in `current.json`. A verdict describes the plan content it reviewed, so any `--canonical-plan` re-sync clears the standing verdict: a revised plan re-enters the optional review loop instead of staying blocked by the old `needs_correction`. Product direction has no review skill: the user approves SPEC.md at frame's exit.

| Review | Verdict | Next skill |
| --- | --- | --- |
| `auto-eng-review` | `approved` | `auto-execute` |
| `auto-eng-review` | `approved_with_risks` | `auto-execute` (risks surfaced before each slice) |
| `auto-eng-review` | `needs_correction` | `auto-plan` |

## STOP Conditions

Halt and report when:

- `canonical_spec` is required but missing or unreadable.
- `canonical_plan` is required but missing or unreadable.
- `canonical_design` is set but the file is missing. Report it and continue only when the active skill says DESIGN.md is optional.
- A stage is asked to consume a future-stage artifact.
- The requested work would add archive behavior, runtime lifecycle enforcement, daemons, dashboards, browser workflows, marketplace behavior, or vendor-source imports without a new SPEC.

## Artifact Signal Discipline

The five signal rules and the deletion test live in `FRAMEWORK.md` (Artifact Signal Discipline). Apply them to every artifact write.
