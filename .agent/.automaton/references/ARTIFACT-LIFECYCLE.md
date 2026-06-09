# Artifact Lifecycle

Shared contract for what each Automaton stage consumes, writes, records, and hands off. This reference guides skills; it does not add runtime enforcement.

## Invariants

Stage list, state contract, and `sync-status.mjs` mandate are in `FRAMEWORK.md`.

- Concrete paths belong in `current.json`, `SPEC.md`, and `PLAN.md`; do not create a separate status prose artifact to mirror them.
- Skills write artifacts only for the active change unless a skill explicitly documents a steering or wiki output.
- Do not add archive behavior here: no archive commands, runtime enforcement, daemons, dashboards, browser workflows, marketplace behavior, or vendor-source imports.

## Progressive Disclosure

`SPEC.md` and `PLAN.md` are canonical indexes, not forced compression targets. For large coherent work, keep canonical files reloadable and link normative detail files instead of narrowing the goal.

Allowed active-change layout:

```text
.agent/work/<change>/INTAKE.md
.agent/work/<change>/SPEC.md
.agent/work/<change>/spec/*.md
.agent/work/<change>/PLAN.md
.agent/work/<change>/slices/*.md
.agent/work/<change>/DESIGN.md
.agent/work/<change>/orchestration/*.md   # conditional: subagent route or complex review loops only
```

- `INTAKE.md` preserves approved office-hours context for `auto-frame`. It is discovered by `active_change`, not by a canonical pointer.
- `SPEC.md` must summarize and link every normative `spec/*.md` detail file. Unlinked supplemental files are notes, not contract.
- `PLAN.md` must link any `slices/*.md` detail file and preserve requirement IDs, gap IDs, invariants, audit questions, migration checkpoints, or coverage targets from SPEC.md.
- Execute and verify load only detail files linked to the active slice or requirement IDs.
- Execute writes slice evidence in place: inline slices update `PLAN.md`; linked detail slices update `slices/slice-NNN.md`; `orchestration/*.md` is supporting evidence, not the default write target.
- Split a change only for independent outcomes. Do not split or narrow one coherent outcome solely because the spec or plan has many files, gaps, constraints, or scenarios.
- If a skill narrows the user's stated scope, it must name the narrowing, explain why, and then widen the scope, ask the user to confirm, or record the deferred scope as a `Deferred / Not in scope` note inside the current change's SPEC. A narrowed scope must not be promoted into a `ROADMAP.md` phase: roadmap phases come only from a user-approved `auto-office-hours` decomposition, never as a side effect of a skill writing a smaller spec.

## Stage Handoffs

| Stage | Required inputs | Produces | State pointer expectations | Next handoff |
| --- | --- | --- | --- | --- |
| `frame` | active change; optional `INTAKE.md` or framing context | `INTAKE.md`, `SPEC.md`, and roadmap update when office-hours approves roadmap scale | office-hours sets `active_change` and `stage: frame`; frame sets `canonical_spec`; `stage` stays `frame` unless plan handoff is approved | **Continue** → `auto-plan` (construction) or `auto-office-hours` (not frameable). **Stop** → `Next: auto-ceo-review` (optional review). |
| `plan` | `canonical_spec`; optional review sections | `.agent/work/<change>/PLAN.md`; optional `DESIGN.md` | `canonical_plan` points to PLAN.md; `canonical_design` only when DESIGN.md exists; `stage` becomes `plan` | **Stop** → `Next: auto-eng-review` (optional) or `Next: auto-execute`. |
| `execute` | approved PLAN.md, current slice, acceptance criteria, verification commands | code/docs/tests plus PLAN-required slice evidence | auto-execute sets `stage: execute` after `canonical_plan` resolves and before changes; do not change canonical pointers to missing files; do not add slice cursor state | **Continue** → re-enter for remaining slices, then `auto-verify` when all complete; **stop** at a valid checkpoint, STOP condition, context pressure, or host limit. |
| `verify` | canonical PLAN.md, executed slices, verification commands | verification report; `VERIFY-GAP` annotations in PLAN.md on failure | auto-verify sets `stage: verify` after `canonical_plan` resolves and before commands; failure returns state to `stage: execute` | **Stop** → `verified` on pass (terminal); `Next: auto-execute` on fail. |
| `verified` | canonical PLAN.md and verification evidence | completed change summary; roadmap phase marked done when applicable | `stage: verified` set only on full verification pass | None. Terminal. `auto-office-hours` only as a new-objective entry point. |
| `resume` | current state and canonical artifact pointers | concise recovery summary and next recommended skill | does not invent missing pointers; stale pointers are reported, not silently repaired | Orient and stop → `Next: <skill matching recovered state>`. |

## Handoff Contract

The two-move model (**Continue inline** / **Stop and hand off**) is in `FRAMEWORK.md`. Continue inline by default so a clean handoff does not force the user to re-invoke the next skill. This is not nested skill invocation (DD-003): no skill calls another; the agent loads the next stage's SKILL.md and proceeds. Do not invent a universal Skill tool or hidden dispatcher.

**Stop and hand off at three edges:**

1. **Entry into `execute`** -> code and project artifacts start changing there, so a human authorizes it. Covers `plan → execute`, `auto-eng-review → execute`, and a failed `verify → execute`.
2. **Entry into an optional review** -> `auto-ceo-review` and `auto-eng-review` are user-invoked. A producing skill recommends a review and stops. It does not auto-run a review on the artifact it just wrote, which would trap the review in the producer's own context.
3. **Verify outcomes** -> a pass closes the change, a fail returns to execute. Stop either way.

`auto-verify` is the mandatory gate, not an optional review, so `execute → verify` continues inline. The audit re-derives from fresh command output, never from execute's reasoning. `auto-onboard` and `auto-resume` are utilities: they report findings and recommend a next skill rather than continuing, so the user keeps the direction. `stage: verified` is terminal. Any `auto-office-hours` mention is for a new objective, not a same-change handoff.

Each handoff carries five durable elements:

1. **Exit gate** -> condition required to advance.
2. **Artifacts produced or updated** -> files written for the active change.
3. **State mutation** -> `current.json` fields changed through `sync-status.mjs`: `stage`, canonical pointers, or review verdicts.
4. **Diagnostic handling** -> `error` diagnostics block advancement. `warning` diagnostics surface to the next stage.
5. **Next-stage recommendation, blocker, or completion note** -> what to invoke next, what blocks progress, or that the active change is complete.

## Checkpoint Semantics

`Checkpoint after:` marks a slice that must pause for human input before the next slice starts. The label vocabulary is pinned in `contracts-data.json`; each value's meaning is defined here, once, so `auto-plan` (which assigns checkpoints) and `auto-execute` (which honors them) cannot drift.

- **`none`** (default) -> no pause. The next slice may start once verification passes.
- **`human-verify`** -> valid only when available commands, tests, host tools, and local inspection cannot verify the result. If any of those can confirm it, it is not a checkpoint.
- **`decision`** -> valid only when a human must choose among named product, architecture, design, scope, or risk options before the next slice can start, and the answer changes that next slice. The checkpoint reason must state the concrete question and the options. Not for reversible engineering judgment, known limitations, verification findings, or "the next slice should be…" notes.
- **`human-action`** -> valid only when progress requires an external action the agent cannot perform, such as 2FA, account approval, or off-machine access.

Verification findings, implementation caveats, downstream consequences, and recommendations for an already-approved next slice are not checkpoints. Record them as slice evidence or risks and continue.

## Git Rhythm

Per-slice commits are owned by `auto-execute`. This reference pins the contract so the skill prompts cannot drift.

**Single owner.** `auto-execute` runs every `git commit` Automaton produces. `auto-verify` never invokes any git write command. Its read-only-on-code gate extends to git history. Subagents on the implementer route never invoke any git command. The orchestrator owns history.

**Trigger.** The rhythm is active when, at execute-stage entry, the working directory is a git repo, the run has no "skip git" instruction, and the repo is not in an interrupted state (mid-rebase, mid-merge, mid-cherry-pick, mid-bisect, or detached HEAD). When active, a commit fires after each slice's verification passes. The verification gate is the authorization, no separate prompt.

**Commit shape.** `git add -A` followed by `git commit -m "slice N: <objective>"` for fresh slices, or `slice N gap-fix: <fix objective>` for gap-fix slices re-entered after `auto-verify` FAIL. Add scope defers to the user's `.gitignore`; the harness does not curate paths.

**Strictly additive.** `git commit` only. Never `amend`, `reset`, `rebase`, `branch`, `checkout`, or `push`. The harness never rewrites history a user might already have inspected.

**Pre-existing dirt.** If the working tree is already dirty at execute-stage entry, `auto-execute` announces once that slice 1's commit will sweep in the pre-existing changes, then proceeds. No question is asked; recovery (`git reset HEAD~`) is in the user's normal toolkit.

**auto-verify leftovers.** Markdown writes from `auto-verify` are not committed by the producing skill. `VERIFY-GAP` blocks added to PLAN.md on FAIL fold into the next `auto-execute` gap-fix commit on re-entry. The one-line ROADMAP phase update on PASS sits in the working tree as a terminal-state note; the user closes it in their own cadence.

**Rhythm STOP conditions.** Commit failure (pre-commit hook rejection, signing failure, etc.) or the repo entering an interrupted state mid-run is STOP-and-surface. The agent does not silently skip the rhythm to keep going, and does not retry with workarounds.

Validation tier: L3 (prompt prose plus `tests/skills.test.mjs` regression). No runtime enforcement; the rhythm is portable across hosts because it lives entirely in skill prompts.

## Review Verdict Routing

`auto-ceo-review` and `auto-eng-review` are optional lifecycle checks, not stage prerequisites. Use them when product direction or execution safety needs review. Downstream skills must respect any review verdict in `current.json`.

Product review may descope or re-scope; engineering review blocks execution safety only.

| Review | Verdict | Next skill |
| --- | --- | --- |
| `auto-ceo-review` | `approved` | `auto-plan` |
| `auto-ceo-review` | `approved_with_risks` | `auto-plan` (risks must appear in plan) |
| `auto-ceo-review` | `needs_clarification` | `auto-frame` or `auto-office-hours` |
| `auto-ceo-review` | `descoped` | `auto-office-hours` or stop |
| `auto-eng-review` | `approved` | `auto-execute` |
| `auto-eng-review` | `approved_with_risks` | `auto-execute` (risks surfaced before each slice) |
| `auto-eng-review` | `needs_correction` | `auto-plan` |

## STOP Conditions

Halt and report when:

- `canonical_spec` is required but missing or unreadable.
- `canonical_plan` is required but missing or unreadable.
- `canonical_design` is set but the file is missing; report it and continue only when the active skill says DESIGN.md is optional.
- A stage is asked to consume a future-stage artifact.
- The requested work would add archive behavior, runtime lifecycle enforcement, daemons, dashboards, browser workflows, marketplace behavior, or vendor-source imports without a new SPEC.

## Validation Tiers

Validation has three tiers. Keep each check at the lowest tier that catches the failure; do not promote artifact-shape or norm checks into runtime.

| Tier | Scope | Enforced by | Example |
| --- | --- | --- | --- |
| **L1 Coordination** | Cross-skill state invariants | `runtime/lib/validate.mjs`; `error`-level diagnostic; hard stop | Stage enum, canonical pointer resolves to an existing file |
| **L2 Artifact shape** | A single artifact's downstream consumability | Next skill reads upstream artifact and surfaces a `warning`-level diagnostic | SPEC.md has Acceptance Criteria; PLAN.md slices have verification commands |
| **L3 Norms** | Wording, structure, prose quality | Prompt text + `tests/skills.test.mjs` regression coverage | Bounded goal is one sentence; lifecycle skills avoid mandatory nested invocation |

Runtime stays portable across Claude, Codex, and OpenCode by holding only L1 checks. L2 lives where artifacts are consumed. L3 lives in prompts and regression tests.

## Artifact Signal Discipline

Automaton artifacts are read by future skills and humans. Every section must change a downstream decision.

1. **No mirror sections** -> One concept per section. If two sections answer the same question, delete one or reframe them.
2. **Index over transcript** -> Aggregate tables (traceability, verification rollups, slice summaries) earn their place only at ≥ 3 entries. For 1–2 entries, inline the information where it is used.
3. **Core versus conditional sections** -> Lifecycle SKILL.md required-section lists distinguish core (always present) from conditional (include only when the named trigger applies). Each conditional section names its trigger.
4. **Append-replace, not stack** -> Review sections on artifacts are replaced on re-run for the same change, not stacked. Do not accumulate multiple `## Review: Product` or `## Review: Engineering` blocks.
5. **Inline default for transient reports** -> Verification reports, status summaries, and intermediate audit output live in the conversation only. Write to disk only when a future skill or human will read it again.

**Deletion test for any section:** if this section were removed, what downstream skill or human loses information? If nothing, drop it.
