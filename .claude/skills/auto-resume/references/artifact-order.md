# Artifact Dependency Order

Artifacts must be loaded in dependency order. A downstream artifact assumes upstream artifacts are already understood.

This table is the orientation load order for `auto-resume` and for cold re-entry. Stage skills own their working loads: what a stage reads while doing its job is governed by that skill's own contract, not by this table.

## Loading Rules by Stage

| Stage | Load These Artifacts | Stop Here |
|-------|----------------------|-----------|
| `frame` | SPEC.md | Do not load DESIGN.md or PLAN.md |
| `plan` | SPEC.md, DESIGN.md (if exists), PLAN.md | Do not load source files |
| `execute` | SPEC.md, DESIGN.md (if exists), PLAN.md, current slice | Do not load unrelated slices |
| `verify` | SPEC.md, PLAN.md, verification evidence | Resume or re-run verification. Spec first, the criteria trace to it |
| `verified` | PLAN.md (change complete; surface pending roadmap items only as context) | Do not reload the full artifact chain or route to new work unless the user asks |

Do not rebuild project context by scanning the repo. The artifacts hold the decisions, and a scan re-derives at full cost what the chain above already records.
