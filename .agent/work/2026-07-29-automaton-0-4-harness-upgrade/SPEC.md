# SPEC: Automaton 0.4 repository harness upgrade

**Bet:** A portable, versioned Automaton harness with one project instruction
source is more reliable than duplicated steering documents and host-local
configuration.

## Bounded goal

Finish the pending Automaton 0.4.0 upgrade so `AGENTS.md` is the sole project
instruction source, stale steering content is removed, and the Codex and Claude
integrations work from any checkout.

## Work scale and shape

- **Scale:** Bug-sized repository maintenance.
- **Shape:** Migration and cleanup of the installed workflow harness.

## Selected lenses

`engineering` for lifecycle correctness and `runtime` for hook portability.

## Constraints and risks

- Preserve the installed Automaton 0.4.0 runtime, skills, agents, and install
  receipt as one coherent versioned upgrade.
- Remove `.agent/steering/PROJECT.md` and
  `.agent/steering/REQUIREMENTS.md`; their content is stale or duplicated and
  must not be migrated elsewhere.
- Keep `.agent/steering/ROADMAP.md` only in Automaton's canonical empty shape.
  `AGENTS.md` must not direct agents to read it.
- Remove stale snapshot material from `AGENTS.md`; retain lasting project rules,
  lifecycle instructions, architecture constraints, and testing requirements.
- Hook commands must not contain user-home, checkout, or version-manager paths.
- Update `.agent/.automaton/state/current.json` only through
  `sync-status.mjs`.
- Preserve the completed Nemotron artifacts and product behavior.
- The existing 174-path harness diff belongs to the user and must be integrated,
  not reverted or mixed with product changes.

## Required outcome

The repository contains one portable Automaton 0.4.0 installation shared by
Codex and Claude. `AGENTS.md` carries durable project instructions, the two
deprecated steering files are absent, the roadmap is empty, terminal state is
recorded through the current lifecycle contract, and the complete harness
upgrade is committed separately from model code.

## Acceptance criteria

1. `.agent/steering/PROJECT.md` and
   `.agent/steering/REQUIREMENTS.md` do not exist.
2. `.agent/steering/ROADMAP.md` exactly follows the empty-roadmap shape in
   `ROADMAP-CONTRACT.md`.
3. `AGENTS.md` contains no reference to the removed steering files or roadmap
   and no model-specific runtime snapshot presented as current project state.
4. Tracked `.agent/`, `.codex/`, and `.claude/` configuration contains no
   `/Users/...` path, credential, or other host-local value.
5. The Codex and Claude skills and agent definitions reflect the same Automaton
   0.4.0 contracts, with differences limited to their host interfaces.
6. `install-manifest.json` is tracked and identifies Automaton 0.4.0.
7. Automaton scripts and both session hooks pass Node syntax checks; both hooks
   run successfully; `get-context.mjs` resolves the canonical state without an
   error diagnostic.
8. `git diff --check` and `pytest tests/unit/` pass.
9. The harness upgrade is committed as a dedicated repository-maintenance
   change with no product-source modifications.

## Anti-goals

- No speech-model, inference, checkpoint, dependency, or public API changes.
- No preservation of stale project or requirements prose in a new location.
- No speculative model roadmap or replacement planning document.
- No hand-editing `current.json`.

## Scope coverage

- **Included:** Automaton runtime upgrade, Codex and Claude integrations,
  portable hooks, stale steering removal, empty roadmap, `AGENTS.md` cleanup,
  lifecycle-state reconciliation, validation, and a dedicated commit.
- **Anti-goal:** Product work and new roadmap selection remain outside this
  maintenance change.
