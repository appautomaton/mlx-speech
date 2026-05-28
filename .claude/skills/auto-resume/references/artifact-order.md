# Artifact Dependency Order

Artifacts must be loaded in dependency order. A downstream artifact assumes upstream artifacts are already understood.

## Dependency Graph

```
REPO-MAP.md (wiki)
    │
    ▼
PROJECT.md (steering)
    │
    ▼
REQUIREMENTS.md (steering)
    │
    ▼
ROADMAP.md (steering)
    │
    ▼
SPEC.md (work) ────────┐
    │                  │
    ▼                  │
DESIGN.md (work)       │
    │                  │
    ▼                  │
PLAN.md (work)         │
                       │
current.json (state) ──┘
```

## Loading Rules by Stage

| Stage | Load These Artifacts | Stop Here |
|-------|----------------------|-----------|
| `frame` | SPEC.md | Do not load DESIGN.md or PLAN.md |
| `plan` | SPEC.md, DESIGN.md (if exists), PLAN.md | Do not load source files |
| `execute` | SPEC.md, DESIGN.md (if exists), PLAN.md, current slice | Do not load unrelated slices |
| `verify` | PLAN.md, verification evidence | Resume or re-run verification |
| `verified` | PLAN.md (change complete; surface pending roadmap items only as context) | Do not reload the full artifact chain or route to new work unless the user asks |
| `resume` | current.json, then canonical artifacts that resolve | Load only what is needed to orient |

## Anti-Patterns

- **Loading PLAN.md before SPEC.md.** The plan assumes the spec is understood.
- **Reloading the full chain at verified stage.** Verification passed; report completion and surface optional future work only when useful.
- **Loading the full wiki during execution.** Wiki pages are reference material, not active context.
