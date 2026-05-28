# Topology Scan

Use the smallest import that can rebuild trustworthy project truth.

## Principle

Do not force the repo into preset shapes like `Node app`, `Python service`, or `frontend/backend split` before reading it.

Instead:

- detect the repo's real control surfaces
- read the smallest representative slice of each surface
- deepen only where the evidence is still insufficient
- stop once `REPO-MAP.md` can explain the repo with clear confidence boundaries

Wild repos are normal. Explain the topology. Do not classify for the sake of comfort.

## Read Order

1. Existing Automaton truth
   - `.agent/.automaton/state/current.json`
   - `.agent/steering/*.md`
   - active `.agent/work/<change>/...`
2. Intent surfaces
   - `README*`, `AGENTS.md`, `CLAUDE.md`, top-level `docs/`
3. Shape surfaces
   - workspace manifests, lockfiles, package roots, service roots, generated-vs-source boundaries
4. Execution surfaces
   - entrypoints, CLIs, routers, workers, jobs, schedulers, migrations, scripts
5. Verification surfaces
   - test config, CI files, lint, typecheck, validation scripts
6. Delivery and integration surfaces
   - deployment config, containers, infra, env templates, external APIs, queues, storage, plugin contracts, event buses
7. Targeted source reads only when the control surfaces are not enough

## Budget

- Prefer top-level docs and manifests before any broad code search.
- Read a representative slice, not the whole repo.
- For each subsystem, read only the files that reveal purpose, boundaries, commands, and integration points.
- Stop expanding the scan once `REPO-MAP.md` can explain the repo without guesswork.

## Output Requirements

Write `.agent/wiki/REPO-MAP.md` before final steering refresh.

That map should capture:

- repo purpose
- topology and runtime surfaces
- apps, packages, services, or other control surfaces
- dev, build, and test commands
- conventions and constraints
- key unknowns and sources read

If a follow-up is needed, use `examples/question-patterns.md`.
