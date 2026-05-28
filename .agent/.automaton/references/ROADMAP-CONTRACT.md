# Roadmap Contract

Compact format and update rules for `.agent/steering/ROADMAP.md`. Load when writing, updating, or reading roadmap phases.

## Empty Roadmap Shape

Use when no active or pending roadmap remains:

```md
# Roadmap

No active roadmap.

First-time onboarding does not create roadmap phases. Refresh imports require strong roadmap evidence and user confirmation in chat.

## Deferred or Not Now

- None recorded.
```

Reset to this shape instead of deleting ROADMAP.md.

## Canonical Phase Format

```
## Phase N: [Name]

- status: pending | active | done
- change: `<change-slug>` | (empty when unframed)
- objective: [bounded outcome]
- why now: [dependency or leverage justification]
- likely outputs: [deliverables]
- evidence: `[file path or command]` | user-stated
- exit signal: [how to verify the phase is complete]
```

Field order is normative. `status` and `change` appear first.

## Status Values

| Status | Meaning | Set by |
|--------|---------|--------|
| `pending` | Queued for future work | `auto-onboard` (confirmed refresh/import only), `auto-office-hours` (user-approved decomposition) |
| `active` | Current framed/planned/executed phase | `auto-office-hours` for the first spec in an approved decomposition |
| `done` | Verified complete | `auto-verify` |

Status progression is one-directional: `pending` → `active` → `done`. Do not reverse.

## Update Rules

| Skill | Action | When |
|-------|--------|------|
| `auto-onboard` | Empty shape on first setup; confirmed refresh may import `pending` phases only | First-time setup or targeted refresh |
| `auto-office-hours` | Replaces content with approved decomposition; first spec is `status: active` with its `change:` slug; may reset inactive roadmap first | Roadmap-sized work and a user-approved phased decomposition |
| `auto-frame` | Does not create roadmap phases; records narrowed scope as a `Deferred / Not in scope` note in the change's SPEC | SPEC is narrower than the user's stated goal |
| `auto-verify` | Marks matching phase `status: done`; resets to empty shape if no active/pending phases or deferred items remain | Final slice passes all criteria |
| `auto-resume` | Reads pending items as context during re-entry or recovery | Resume, compaction, stale state, or explicit recovery |

## Matching Rule

`auto-verify` matches a roadmap phase to the active change by comparing the phase's `change:` field to `active_change` in `current.json`. If `change:` is empty or does not match, skip the roadmap update.

## Invariants

- There is exactly one roadmap file: `.agent/steering/ROADMAP.md`. Do not create parallel roadmap files.
- ROADMAP.md is a steering artifact. It is NOT a canonical pointer in `current.json`.
- ROADMAP.md is forward-looking. Work evidence lives in `.agent/work/<change>/`; ROADMAP.md is not a completed-work history log.
- A user-approved `auto-office-hours` roadmap replaces existing roadmap content and supersedes a speculative onboard roadmap.
- `auto-onboard` must not create roadmap phases during first-time onboarding.
- `auto-onboard` must not synthesize roadmap phases from repo evidence alone during refresh.
- `auto-onboard` must not create `status: active` phases.
- At most one phase has `status: active` at any time.
- A phase with `status: active` must have a non-empty `change:` field.
- `auto-frame` does not create roadmap phases. When a SPEC is narrower than the user's goal, frame records the deferred scope as a `Deferred / Not in scope` note in the change's SPEC; phases come only from a user-approved `auto-office-hours` decomposition.
- The `## Deferred or Not Now` section at the bottom holds items explicitly excluded from the roadmap.
- Inactive means every phase is `done` and `## Deferred or Not Now` is empty or `None recorded`; writer skills may reset inactive roadmaps to the empty shape.

## Anti-Patterns

- Creating parallel roadmap files (e.g., `ROADMAP-<name>.md`) instead of updating `ROADMAP.md`.
- Creating roadmap phases during first-time onboarding.
- Using onboarding to create speculative roadmap phases without user confirmation.
- Using onboarding to create active roadmap phases.
- Adding ROADMAP.md as a canonical pointer in `current.json`.
- Setting multiple phases to `status: active` simultaneously.
- Skipping `pending` and writing phases directly as `active` without user approval.
- Reversing status (e.g., `done` back to `active`).
- Preserving done-only roadmap history after all roadmap items are complete.
- Adding fields to phase format without updating this contract.
