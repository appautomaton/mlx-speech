---
name: auto-office-hours
description: Sharpen a vague idea into a bounded objective. Use before framing when scope is undefined.
metadata:
  stage: frame
---

# auto-office-hours

Pre-frame conversation. Turns a vague idea into a sharp objective before framing begins.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-office-hours owns clarity before framing: classify the work internally, test the missing assumptions, preserve request coverage, present approaches, and write approved intake. It does not write code or scaffold projects. It does not create SPEC.md in conversational mode. Before approval, it writes nothing.

Loading discipline: keep the conversation goal, evidence, request coverage, rejected framings, and next decision in context. Read project files only when repo evidence changes the objective, especially for parity, audit, migration, coverage, or mixed work. When repo evidence would otherwise pull wide reads into context, you may dispatch the read-only `automaton-librarian` for a one-shot lookup (see `.agent/.automaton/references/LIBRARIAN.md`); it returns evidence, you keep the decision.

Interaction: keep chat plain, organized, and grounded in the user's words. Do not expose taxonomy labels such as mode, scale, or shape. For real branch decisions, offer 2–4 concrete options with a one-line reason for each. Use the host question tool when available; otherwise present the same options inline.

## Quality Gate

Before presenting alternatives, recommending an approach, or writing `INTAKE.md`:
- Confirm request coverage before narrowing scope or deferring work.
- Verify the objective reflects the user's current intent, not the initial framing.
- Make alternatives differ by scope, risk, learning value, traceability, or verification strength.
- Evaluate evidence directly: name what is supported, what is missing, and what evidence would change the assessment.
- Read `references/quality.md` when the conversation sounds encouraging but non-decisive.

## Do

### Classify The Work

Determine three internal axes:
- **Mode:** Startup mode for customers, revenue, market, competition, fundraising, or company-building; Builder mode for side project, hackathon, learning, open source, personal use, or just-for-fun; Content mode for writing, article, brief, deck, blog post, newsletter, documentation, or any prose where audience and voice matter.
- **Work scale:** bug-sized, feature-sized, capability-sized, or roadmap-sized. Do not equate "large" with roadmap-sized. Capability-sized work remains one spec when it serves one coherent outcome; roadmap-sized means multiple independently valuable outcomes that need decomposition through `ROADMAP.md`.
- **Work shape:** feature, refactor, parity, audit, migration, coverage, content, or mixed.

Hold this classification internally to steer questioning. Confirm the read in plain language grounded in the user's words, not by naming the taxonomy. If the user corrects any dimension, adjust before continuing. For bug-sized goals with a known fix, consider whether `auto-frame` is the better entry point. For Startup or Builder mode, read `references/operating-principles.md` for doctrine; for Content mode, read `references/content-intake.md`; for roadmap-sized goals, read `.agent/.automaton/references/ROADMAP-CONTRACT.md`.

### Run Diagnostic

Ask only questions that make the objective frameable. Use the active reference:
- Startup Mode: read `references/startup-diagnostic.md` when demand, user, market, or customer evidence matters; read `references/landscape-awareness.md` when market, ecosystem, competitor, or current-state evidence would change the frame.
- Builder Mode: read `references/builder-diagnostic.md` when the work is personal, exploratory, open-source, or design-partner shaped.
- Content Mode: read `references/content-intake.md` when the deliverable is writing, article, brief, deck, newsletter, documentation, or other prose.

When the shape is not feature, shape-specific questions take priority: parity needs a reference system and gap-closure target; audit needs questions and decision use; refactor needs invariants and blast radius; migration needs source/target state and rollback; coverage needs risk areas and verification target; mixed work needs the highest-priority question from each shape.

Follow up when an answer changes scope, reveals a constraint, contradicts earlier context, or stays abstract. Ask for a concrete correction or choice, not a generic reaction. If the answer is polished but vague, push until it names concrete evidence, a specific stakeholder, or an observable workaround. Read `references/diagnostic-calibration.md` when the diagnostic feels soft or agreeable rather than evidence-backed.

### Request Coverage

Before generating alternatives, build a compact coverage map from the user's request and answers: goal, context/background, perspectives or audiences, constraints, worries/risks, explicit asks, and implied asks.

Classify each material item as:
- **Included** in the current change.
- **Deferred** to later work, with the reason, recorded in `INTAKE.md`. Promote to `ROADMAP.md` only when the user approves a phased decomposition.
- **Anti-goal** for this change.
- **Needs decision** because the answer would change scope, approach, or verification.

If any item would be narrowed or dropped, name the reason. If a decision is needed, ask one focused question or offer 2–3 concrete options before recommending an approach. Keep this as a decision map, not a transcript.

### Generate Alternatives

Present 2–3 distinct approaches that match the user's scale and shape. Include a minimal viable option and an ideal architecture option for bug, feature, and capability work; for roadmap-sized work, offer decomposition strategies or first-spec candidates. For refactor, parity, audit, migration, or coverage, differentiate by blast radius, traceability, evidence depth, rollout risk, or verification strength. Read `references/alternatives-format.md` for the exact format.

Recommend one approach and explain what evidence supports it, what it does not prove, and what evidence would change the recommendation. Do not proceed until the user explicitly approves an approach or chooses a different one.

### Persist Approved Intake

After approval, derive a date-prefixed change slug: `YYYY-MM-DD-<kebab-case-objective>` using today's date. Reuse `active_change` only when it already matches this discussion. Write `.agent/work/<change>/INTAKE.md` using `references/intake-template.md`; Content mode includes the required content fields from `references/content-intake.md`.

When scale is roadmap and the user has approved a phased decomposition, replace `.agent/steering/ROADMAP.md` with that approved decomposition per `.agent/.automaton/references/ROADMAP-CONTRACT.md`. Without that explicit approval, leave `ROADMAP.md` untouched and keep deferred scope in `INTAKE.md`.

Run `node .agent/.automaton/scripts/sync-status.mjs --active-change "<change>" --stage frame` from the project root. This records `active_change` and `stage` through the shared state validator.

### Continue To Frame When Ready

After `INTAKE.md` is written, continue inline into `auto-frame` when all are true:
- The approved intake states the objective in one sentence.
- Scope coverage has no unresolved `Needs decision` item that would change scope, approach, or verification.
- The target stakeholder or artifact, desired outcome, constraints, anti-goals, and key risks are clear enough to produce acceptance criteria.
- The host/session can write `SPEC.md` without dropping material request context.

If those conditions pass, load and follow `auto-frame`'s contract, write `.agent/work/<change>/SPEC.md`, and let auto-frame record `canonical_spec`. If any condition fails, stop after intake with the blocker or focused question.

<GATE>

Do NOT create INTAKE.md, SPEC.md, DESIGN.md, or implementation artifacts until:
- The user has explicitly approved one presented approach.
- Blocking questions are resolved or explicitly accepted as assumptions.

There are no file writes before the user picks an approach.
</GATE>

<STOP>

Halt and report when the user wants a solution before describing the problem, or when the minimum diagnostic still cannot identify a stakeholder, desired outcome, content audience/thesis, concrete evidence, or observable workaround. Do not guess.
</STOP>

## Output

`INTAKE.md` is guaranteed only for an approved office-hours session; aborted, skipped, or still-conversational sessions do not produce it.

Approved path:
- `INTAKE.md` written to `.agent/work/<change>/INTAKE.md` from the relevant intake template.
- Objective uses the user's final refined wording.
- Scope coverage: included, deferred, anti-goals, and needs-decision items; omit empty groups.
- Scope preservation records whether the intake preserves the full stated intent or intentionally decomposes it.
- Deferred scope is named for later work in `INTAKE.md`; it is promoted to `ROADMAP.md` only on a user-approved phased decomposition.
- `stage: frame` and `active_change` are recorded through `sync-status.mjs`.
- `.agent/steering/ROADMAP.md` is updated only when the user approves a phased decomposition.
- Approved, complete intake continues inline into `auto-frame` without another user prompt when frame-ready.

The INTAKE.md is a decision record, not a transcript. It is a faithful record of what the user approved, not the agent's editorial rewrite.

If the user does not approve an approach, output a short discussion summary, why no approach was selected, deferred scope worth preserving, a `Next:` line, and no file writes.

## Rules

- Do not drop request context silently; every material ask, context detail, perspective, or worry is included, deferred with reason, marked as an anti-goal, or turned into a focused question.
- Ask follow-up questions when they matter; do not bank them for a later checklist.
- State the decision basis; name what evidence supports, what it does not support, and what would change the recommendation.
- Keep INTAKE compact; omit empty sections and analysis nobody downstream needs.
- If the user's language shifts from exploration to urgency, or from technical to business framing, reclassify mode, scale, or shape and state the change in plain language.
- If the user expresses impatience, ask the two most critical unresolved questions; if they push back again, present alternatives with explicit assumptions.
