---
name: auto-onboard
description: Build project truth from repo evidence. Use when steering is missing or stale.
metadata:
  stage: frame
---

# auto-onboard

Repository discovery. Builds bounded project truth from evidence, not guessing.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-onboard builds bounded project truth from repository evidence, not training data, not conversation, not guessing. It does not write code or produce specs. Loading discipline: keep REPO-MAP.md under 150 lines; stop scanning once you have enough.

## Quality Gate

Before writing steering artifacts:
- Separate observed, inferred, and unknown facts.
- Cite paths for repo-shape claims.
- Treat artifact writing as expensive: write only durable project truth and immediate blockers, not scratch notes.
- Stop scanning once the next action is clear.
- Read `references/quality.md` when artifacts turn into broad inventory.

## Do

### Detect State

Choose the smallest valid path:
- First-time or scaffold-level steering: scan and write the starter artifacts.
- Real steering, no refresh requested: do not rewrite; report what exists and route active work to `auto-resume`, otherwise `auto-office-hours`.
- Targeted refresh: read only evidence relevant to the requested steering update and rewrite only affected artifacts.

ROADMAP.md stays restrained. Do not create roadmap phases on a first run. On refresh, write phases only when strong repo evidence shows an existing or ongoing roadmap and the user confirms importing or refreshing it in chat; then use `.agent/.automaton/references/ROADMAP-CONTRACT.md`.

### Scan Top-Level Files

Read `README.md`, `package.json` or equivalent, and up to 3 config files (e.g., `.gitignore`, `tsconfig.json`, `Makefile`). Stop at 5 files.

### Map Topology

Read `references/topology-scan.md` for the scan protocol. Identify:
- Runtime surfaces (CLI, API, UI, worker)
- Package boundaries (apps, packages, modules)
- Stack (language, framework, build tool, test runner)
- Commands that work today (install, build, test, lint)

### Ask (if necessary)

Ask only when ambiguity changes steering output. Read `references/question-patterns.md`; if one targeted repo read can answer it, read instead. For ROADMAP.md, never ask on first-time onboarding; on refresh, ask only when strong roadmap evidence exists but confirmation is missing.

### Write Artifacts

Apply `references/artifact-contract.md` for exact format and required sections. Use `templates/` as scaffolds:
- `.agent/wiki/REPO-MAP.md`: bounded evidence index; no open-question parking, confidence verdict, or recommended next skill
- `.agent/steering/PROJECT.md`: compact identity record; what this repo owns and why
- `.agent/steering/REQUIREMENTS.md`: durable constraints only; no generic unknown parking
- `.agent/steering/ROADMAP.md`: compact placeholder on first run; refresher-only phase updates when strong roadmap evidence exists and the user confirms in chat

### Update State

Do not overwrite an existing `active_change` or `stage`. `current.json` is initialized by install/scaffold when missing; auto-onboard writes steering truth, not active-change state.

### Report

Summarize what you found, what you wrote, and what remains uncertain.

<GATE>

Do NOT proceed past scanning if:
- The repository has no `README.md`, no `package.json` equivalent, and no recognizable directory structure after reading 10 files.
- The user has not confirmed whether to overwrite existing steering artifacts.

If the repo is empty or unrecognizable, report this and stop.
</GATE>

<STOP>

Halt and report when:
- A required file (`README.md`, root config) exists but cannot be parsed.
- The repository contains multiple projects at the same level and you cannot determine which is primary.
- The scan reveals conflicting conventions (e.g., both npm and poetry in the same root) and the user cannot clarify.

Do not guess. Do not proceed.
</STOP>

## Output

- Steering artifacts: `.agent/wiki/REPO-MAP.md`, `.agent/steering/PROJECT.md`, `.agent/steering/REQUIREMENTS.md`, `.agent/steering/ROADMAP.md`
- `.agent/.automaton/state/current.json` is initialized by install/scaffold when missing; auto-onboard does not overwrite an existing `active_change` or `stage`
- Warning-level findings surface to the steering artifacts.
- Orient and stop (utility skill): recommend `auto-office-hours` (when scale or shape is undefined) or `auto-frame` (bounded goal already in hand). auto-onboard reports and stops rather than continuing, so the user picks the direction.

## Rules

- Read no more than 10 files total; summarize, do not transcribe.
- Cite a file path for every steering claim.
- Delete empty template sections; templates are prompts, not required headings.
- Keep durable artifacts free of speculative questions, confidence labels, and routing chatter.
- Never create roadmap phases on first-time onboarding. On refresh, change roadmap phases only when evidence and user confirmation justify it.
