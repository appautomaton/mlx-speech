# PLAN: Automaton 0.4 repository harness upgrade

**Goal:** Complete the approved Automaton 0.4.0 migration in
`SPEC.md` without changing product code.

## Execution routing and topology

- **Default:** direct continuation through the single slice.
- **Checkpoints:** none.
- **Parallel-safe groups:** none.
- **Subagent routes:** none; the integration is one shared configuration change.

## Ordered slice sequence

### Slice 1: Integrate, validate, and commit the harness upgrade

**Objective:** Normalize the installed Automaton 0.4.0 payload into one portable,
self-consistent repository-maintenance commit.

**Acceptance criteria:**

- Remove `.agent/steering/PROJECT.md` and
  `.agent/steering/REQUIREMENTS.md` without relocating their content.
- Reset `.agent/steering/ROADMAP.md` to the canonical empty shape.
- Remove the steering reference and model-specific runtime snapshot from
  `AGENTS.md` while retaining durable project, architecture, process, and test
  rules.
- Replace host-local paths in the Codex and Claude hook configuration with
  checkout-portable commands.
- Retain the complete Automaton 0.4.0 runtime, pruned skills, new skills and
  references, synchronized host adapters, and tracked installation receipt.
- Reconcile lifecycle state exclusively with `sync-status.mjs`.
- Fix whitespace or file-mode defects exposed by validation.
- Commit only harness files, `AGENTS.md`, and this change's Automaton artifacts;
  no product source, model, test, dependency, or public documentation files.

**Verification:**

```bash
test ! -e .agent/steering/PROJECT.md
test ! -e .agent/steering/REQUIREMENTS.md
node -e "const fs=require('fs'); const expected='# Roadmap\n\nNo active roadmap.\n\n## Deferred or Not Now\n\n- None recorded.\n'; if(fs.readFileSync('.agent/steering/ROADMAP.md','utf8')!==expected) process.exit(1)"
rg -n 'PROJECT\.md|REQUIREMENTS\.md|ROADMAP\.md|## Runtime State' AGENTS.md
rg -n '/Users/' .codex/hooks.json .claude/settings.json
node -e "const m=require('./.agent/.automaton/state/install-manifest.json'); if(m.automaton_version!=='0.4.0') process.exit(1)"
node --check .agent/.automaton/lib/context.mjs
node --check .agent/.automaton/lib/steering.mjs
node --check .agent/.automaton/scripts/get-context.mjs
node --check .agent/.automaton/scripts/sync-status.mjs
node --check .claude/hooks/session-start.mjs
node --check .codex/hooks/session-start.mjs
node .claude/hooks/session-start.mjs
node .codex/hooks/session-start.mjs
node .agent/.automaton/scripts/get-context.mjs
diff -qr .codex/skills .claude/skills
git diff --check
pytest tests/unit/
git status --short
git diff --name-only HEAD^ HEAD
```

The two `rg` checks must produce no matches. The skill-tree comparison may list
only the three host-specific `HOST-TOOLS.md` files. The final commit file list
must be limited to `.agent/`, `.codex/`, `.claude/`, and `AGENTS.md`; the final
working tree must be clean.

**Produces:** A dedicated Automaton 0.4.0 harness-upgrade commit and a repository
ready for the next independently framed objective.

**Status:** complete
**Evidence:** Removed both deprecated steering files; reset `ROADMAP.md` to the
canonical empty body; removed stale steering and Moss runtime snapshots from
`AGENTS.md`; made both hook launchers checkout-portable; retained the 0.4.0
install receipt and synchronized Codex/Claude payload. Both hook commands ran,
all Automaton JavaScript passed syntax checks, `get-context.mjs` returned no
diagnostics, skill trees differed only in the three host-specific
`HOST-TOOLS.md` files, credential and out-of-scope scans found nothing, and
`git diff --check` passed. `.venv/bin/python -m pytest tests/unit/`: 582 passed.
**Risks / next:** none.

## Verification

### Summary

**Overall:** PASS
**Passed:** 8 of 8 slice criteria
**Remaining gaps:** none

**Slice 1 — Integrate, validate, and commit the harness upgrade:** PASS, 8
criteria. Fresh checks proved both stale steering files absent and
the roadmap byte-identical to the empty contract; found no stale `AGENTS.md`
references, host paths, credentials, whitespace errors, or out-of-scope commit
paths; confirmed the 0.4.0 receipt, executable state scripts, JavaScript syntax,
portable hook launchers, clean Automaton context, and Codex/Claude parity except
for the three expected host-specific `HOST-TOOLS.md` files. Commit `caaeb9c`
passed `git show --check`; `.venv/bin/python -m pytest tests/unit/` passed 582
tests. The virtual-environment invocation replaced the unavailable bare
`pytest` executable. No checks were skipped. Content inspection confirmed that
`AGENTS.md` only lost stale sections and `ROADMAP.md` uses the normative empty
body, so no new prose or anti-slop pattern was introduced.
