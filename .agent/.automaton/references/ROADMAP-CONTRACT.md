# Roadmap Contract

Compact format and update rules for `.agent/steering/ROADMAP.md`. Load when writing, updating, or reading roadmap phases.

## Empty Roadmap Shape

Use when no active or pending roadmap remains:

```md
# Roadmap

No active roadmap.

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
- exit signal: [how to verify the phase is complete]
```

Field order is normative. `status` and `change` appear first.

## Status Values

| Status | Meaning | Set by |
|--------|---------|--------|
| `pending` | Queued for future work | `auto-frame` (user-approved decomposition) |
| `active` | Current framed/planned/executed phase | `auto-frame`: the first spec in an approved decomposition, or adoption of a pending phase that matches a new approved objective |
| `done` | Verified complete | `auto-verify` |

Status progression is one-directional: `pending` → `active` → `done`. Do not reverse.

## Update Rules

| Skill | Action | When |
|-------|--------|------|
| `auto-frame` | Replaces content with the approved decomposition; first spec is `status: active` with its `change:` slug; may reset an inactive roadmap first | Roadmap-sized work and a user-approved phased decomposition |
| `auto-frame` | Adopts a pending phase: sets `status: active` and writes the change slug into its `change:` field | A new approved objective matches a pending phase |
| `auto-frame` | Writes no phase; records the narrowed scope as a `Deferred / Not in scope` note in the change's SPEC | SPEC is narrower than the user's stated goal |
| `auto-verify` | Marks matching phase `status: done`; resets to empty shape if no active/pending phases or deferred items remain | Final slice passes all criteria |
| `auto-resume` | Reads pending items as context during re-entry or recovery | Resume, compaction, stale state, or explicit recovery |

## Matching Rule

`auto-verify` matches a roadmap phase to the active change by comparing the phase's `change:` field to `active_change` in `current.json`. If `change:` is empty or does not match, skip the roadmap update.

## Invariants

- There is exactly one roadmap file: `.agent/steering/ROADMAP.md`. Do not create parallel roadmap files.
- ROADMAP.md is a steering artifact. It is NOT a canonical pointer in `current.json`.
- ROADMAP.md is forward-looking. Work evidence lives in `.agent/work/<change>/`. ROADMAP.md is not a completed-work history log.
- A user-approved decomposition replaces existing roadmap content.
- Phases come only from a user-approved decomposition. Nothing synthesizes them from repo evidence alone.
- At most one phase has `status: active` at any time.
- A phase with `status: active` must have a non-empty `change:` field.
- A narrowed SPEC never becomes a roadmap phase. Frame records that deferred scope as a `Deferred / Not in scope` note in the change's SPEC. Phases come only from a decomposition the user explicitly approved.
- The `## Deferred or Not Now` section at the bottom holds items explicitly excluded from the roadmap.
- Inactive means every phase is `done` and `## Deferred or Not Now` is empty or `None recorded`. Writer skills may reset inactive roadmaps to the empty shape.
- Do not add fields to the phase format without updating this contract.
