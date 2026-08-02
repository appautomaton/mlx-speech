# Slice Routing and Topology

If you cannot write the verification command before starting the slice, the slice is not well-defined. The slice field template lives in `SKILL.md`. This file covers the two judgment calls it does not: when a slice earns the subagent route, and how the topology section renders.

## Route Assignment

Breadth across subsystems is the signal, not slice size. A long slice inside one file stays direct.

The threshold, made concrete: "migrate session auth to JWT across API routes" earns `subagent recommended` because it touches middleware, two route modules, a shared utility, and their tests, changing an interface all of them depend on. Rewriting one of those files, however long the rewrite, does not.

## Topology Section

A PLAN.md topology section names the default route, then only the overrides:

```markdown
## Execution Routing and Topology

Default: direct, serial, continuation after verification.

Overrides:
- Slice 4: subagent recommended (crosses auth middleware and routing with a shared interface change)

**Parallel-safe groups:**
- Slices 2 and 3 (disjoint write sets: Slice 2 touches `src/db/migrations/`, Slice 3 touches `src/ui/components/`; no shared state)

Checkpoints:
- Slice 6: human-verify (visual layout review, not automatable)
```

Each override states its reason inline. A parallel-safe group names the disjoint write sets that make it safe, since that claim is what the reader has to check.
