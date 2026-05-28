# Onboarding Artifact Contract

`auto-onboard` should create project truth in layers instead of collapsing everything into one monolithic note.

## Progressive Disclosure

1. `.agent/wiki/REPO-MAP.md`
   Import evidence, repo shape, live commands, and ambiguity.
2. `.agent/steering/PROJECT.md`
   Stable project identity. This answers why the repo exists and what it currently owns.
3. `.agent/steering/REQUIREMENTS.md`
   Durable commitments. This captures what must stay true, what the repo already constrains, and what is explicitly out of scope.
4. `.agent/steering/ROADMAP.md`
   Roadmap decision surface. It stays short on first-time onboarding; refreshes need strong roadmap evidence and user confirmation.

The sequence should read like `why -> what must stay true -> what may come next`.

## Writing Standard

- Remove scaffold prompts instead of preserving them.
- Lead with the conclusion, then support it with evidence.
- Prefer short sections with strong headings over long prose.
- Prefer tables and compact lists when the source material is scan-heavy.
- Separate `Observed`, `Inferred`, and `Needs Confirmation` when certainty differs.
- When a user follow-up is needed, ask a bounded decision question instead of outsourcing discovery.
- Do not use durable artifacts as scratchpads for speculative questions, confidence labels, or routing chatter.
- Keep ROADMAP.md compact by default. Do not create phases on first-time onboarding, and do not promote candidate phases during refresh without user confirmation.
- Keep roadmap items evidence-backed and near-term when phases are confirmed. Do not invent distant strategy.
- Name concrete files, packages, and commands whenever they anchor the truth.
- Do not let one artifact duplicate the full content of another. Each artifact should narrow the surface area.
- Do not duplicate canonical artifact paths across steering files. `current.json`, SPEC.md, and PLAN.md own active work pointers.

## Confidence Model

- `Observed`: directly supported by files, commands, or repo structure that were read.
- `Inferred`: likely true from the evidence, but not stated directly.
- `Needs Confirmation`: materially important, but not yet safe to promote to project truth.

## Artifact Expectations

### `REPO-MAP.md`

- capture what was read and why it matters
- explain the repo shape in one pass
- preserve sources, commands, and only steering-blocking ambiguity
- make it easy for later skills to avoid re-scanning the whole repo
- do not include `Open Questions`, `Import Verdict`, steering confidence, or recommended next skill sections
- put routing and next-action guidance in the chat report, not in `REPO-MAP.md`

### `PROJECT.md`

- why the repo exists
- owned runtime surfaces
- stack and key observed commands
- visible decision principles already encoded in the repo that affect future changes
- omit user/operator, dependency, or command sections when REPO-MAP.md already holds the detail and no downstream decision changes

### `REQUIREMENTS.md`

- durable product and technical constraints
- invariants already encoded in the repo that future changes must preserve
- quality and operational expectations that later plans must respect
- non-goals that the current system clearly rejects
- only unknowns that block framing, planning, or verification
- do not carry generic unknowns such as missing CI preference, typecheck preference, or "is this user-facing?" unless the answer changes the active change

### `ROADMAP.md`

- default first-time onboarding to the lightweight placeholder
- never write phases during first-time onboarding
- on refresher runs, write 3 to 6 ordered `pending` phases only when strong repo evidence shows an existing or ongoing roadmap and the user confirms importing or refreshing it in chat
- on refresher runs, if strong roadmap evidence exists but confirmation is missing, ask one bounded follow-up before writing phases
- each confirmed phase must include `status: pending` and an empty `change:` field; auto-onboard never writes `status: active`; see `.agent/.automaton/references/ROADMAP-CONTRACT.md` for the full format
- each confirmed phase should have an objective, why now, likely outputs, and an exit signal
- confirmed phases should reflect the current repo, not generic best practices

## Work Artifact Integrity

Work artifacts under `.agent/work/<change>/` may carry review annotations. These sections are append-only and must survive refreshes.

- Any heading matching `## Review: <Type>` in `SPEC.md`, `PLAN.md`, or `DESIGN.md` is a durable annotation.
- Controllers that refresh a work artifact must preserve existing `## Review:` sections and place them after the main content.
- A review section may be updated in place (e.g., a later review revises the verdict), but it must not be silently dropped.
- Only the user may request consolidation or removal of a review section.

This rule ensures that `auto-ceo-review`, `auto-eng-review`, and any future review gates remain discoverable by `auto-resume` and downstream controllers.

## Failure Mode to Avoid

Do not write elegant fiction. If the repo does not prove something, mark it as inferred or unknown.
