---
name: auto-frame
description: Turn a request into a bounded SPEC.md, with as much conversation as it needs. Use to start a change that spans sessions or needs agreed scope.
metadata:
  stage: frame
---

# auto-frame

Framing controller. Turns a request into one bounded `SPEC.md`, running as much conversation as the request needs and no more.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-frame produces the canonical artifact: `SPEC.md`. No file means no completed frame. It does not write code, create PLAN.md, or proceed to planning.

Depth is chosen after reading, never before: the call needs the repo and the request in hand, so it cannot be made at the door.

Loading discipline: hold the objective, constraints, risks, and source evidence that keep the spec real. Avoid exhaustive tree walks. When a lookup would pull wide reads into context, dispatch the read-only `automaton-librarian` (see `.agent/.automaton/references/LIBRARIAN.md`): it returns evidence, you keep the decision. Never ask what the repo can answer.

Artifact discipline: `SPEC.md` is the reloadable contract, not the whole dossier. Layout and linking rules live in `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Progressive Disclosure).

Interaction: keep chat plain, in the user's words. Do not expose the internal labels below. Follow `.agent/.automaton/references/FRAMEWORK.md` (Asking The User).

## Quality Gate

- Make the objective observable.
- Preserve the user's broader intent. Do not silently narrow scope.
- Move implementation detail out unless it constrains scope.
- Mark uncertain claims as assumptions.
- Read `references/quality.md` before presenting alternatives or finalizing `SPEC.md`.

## Do

### Read The Request

Read the request, the conversation, and enough repo evidence to know what is already true. If a `SPEC.md` exists for this change, read it and preserve every `## Review:` section.

Classify three axes and hold them internally:
- **Mode:** Startup mode for customers, revenue, or market. Builder mode for side projects, learning, or open source. Content mode for writing, article, brief, deck, newsletter, documentation, or any prose where audience and voice matter.
- **Work scale:** bug, feature, capability, or roadmap. Large is not roadmap. Capability-sized work stays one spec when it serves one coherent outcome. Roadmap-sized means multiple independently valuable outcomes that need decomposition.
- **Work shape:** feature, refactor, parity, audit, migration, coverage, content, or mixed.

Confirm the read in plain language grounded in the user's words. If the user corrects a dimension, adjust before continuing.

Read `.agent/steering/ROADMAP.md` when it exists. If the objective matches a pending phase, say so and scope the work around that phase.

### Check Engagement

Apply the engagement criterion from the session reminder here, where the request is read and nothing else is spent yet. Work this session can finish and verify does not need a spec: say so in one line and do it directly, because a spec that only restates a request is a record nobody reads again.

The user naming a stage, or asking for a spec, settles this. Frame.

### Choose Depth

State the goal in one sentence, then pick the path and say which you took in one line.

Name what the request leaves open: a problem, a stakeholder, a desired outcome, a content audience and thesis, a first independent outcome, or a direction choice. Route on how many questions would resolve it.

- None: write the spec now.
- One or two: ask, then write. An offer costs more turns than the questions do.
- Three or more, high-stakes (auth, schema, concurrency, migration, payments), or roadmap-sized: offer the depth choice. Roadmap-sized work always earns it: decomposition is the user's decision, not a side effect of writing a smaller spec.

Name the quick pass's question count in the offer, so the choice is between known costs:

- **Quick pass (Recommended):** only what would change scope, approach, or verification.
- **Grill me:** every branch of the decision tree, in dependency order, until it resolves.

A user who already asked for a grill gets one. Skip the offer.

Do not run a diagnostic to look thorough, and do not skip one to look fast.

Before any questioning path, read `references/diagnostic.md`: it carries the mode diagnostics, grill mode, and the alternatives contract.

### Name The Change

If `active_change` is `bootstrap` or does not match the current objective, derive a new slug: `YYYY-MM-DD-<kebab-case-objective>` using today's date. Derive it now: the ROADMAP adoption in Cover The Request and the SPEC.md write both use it. Recording a new change over an unfinished one follows the parking rule in `.agent/.automaton/references/FRAMEWORK.md` (State Contract).

### Cover The Request

Build a compact map from the request and any answers: goal, context, perspectives or audiences, constraints, worries, explicit asks, implied asks. Classify each material item as **included** in this change, **deferred** to later work with a reason, an **anti-goal** for this change, or **needs decision** because the answer would change scope, approach, or verification.

Every included item lands in the bounded goal, required outcome, constraints, risks, or acceptance criteria. Every deferred item keeps its reason in a `Deferred / Not in scope` note. Every anti-goal appears in SPEC anti-goals. Every needs-decision item gets one focused question with concrete options and your recommended answer, per the Asking The User convention, unless the user explicitly accepts an assumption. Do not drop a material item silently.

If your SPEC would be narrower than the user's stated goal, widen it, ask for confirmation, or record the deferred scope. The rule's home is `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Progressive Disclosure). Silent narrowing is a framing failure. A narrowed SPEC never becomes a `ROADMAP.md` phase.

Keep this a decision map, not a transcript.

Roadmap phases come only from a decomposition the user has approved. When they approve one for roadmap-scale work, replace `.agent/steering/ROADMAP.md` per `.agent/.automaton/references/ROADMAP-CONTRACT.md`. Without that approval, leave `ROADMAP.md` untouched and keep the deferred scope in the SPEC. When the approved objective matches a pending phase, adopt it: set `status: active` and write the change slug into its `change:` field.

### Surface

List only constraints, unknowns, and risks that change implementation or verification. Keep decision-critical material in `SPEC.md`. Link larger coherent detail under `spec/*.md`. If constraints point to unrelated outcomes, ask which outcome to frame first.

### Select Lenses

Choose the minimum useful lenses from `product`, `engineering`, `design`, `security`, `runtime`, and `content`. Default to `product` + `engineering`: even pure-engineering changes carry product risk, and a minimal set keeps the spec focused. Add `security` from the start when the change touches auth, data, or trust. For content work, add the content lens; the lens set lives in `references/content-framing.md` (Lens Interaction).

### Write SPEC.md

<GATE>

Do NOT write `SPEC.md` while a decision that would change scope, approach, or verification is unresolved. Resolve it, or get the user's explicit acceptance of an assumption, first. When you presented alternatives, that means the user picked one: presenting options and then writing your own recommendation is not approval.

Do NOT finish framing without `SPEC.md` at `.agent/work/<change>/SPEC.md`.
</GATE>

Read `references/spec-shape.md`. Write its **core** fields, and its **conditional** fields only when their named trigger applies. Apply Artifact Signal Discipline from `.agent/.automaton/references/FRAMEWORK.md` while writing. For large coherent work, follow `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Progressive Disclosure).

The spec is a decision record, not a transcript. It records what the user approved, in the user's final refined wording, not your editorial rewrite.

### Update State

After writing SPEC.md, run `node .agent/.automaton/scripts/sync-status.mjs --active-change "<change>" --canonical-spec ".agent/work/<change>/SPEC.md" --stage frame` from the project root. auto-plan owns the `stage: plan` mutation and records it when it writes PLAN.md, including on inline continuation.

<STOP>

Halt and report when the user wants a solution before describing the problem, or when the diagnostic still cannot identify a stakeholder, desired outcome, content audience and thesis, concrete evidence, or observable workaround. Do not guess.
</STOP>

### Hand Off

Frame's exit is a mandatory stop. Do not plan in the same turn. The edge's why: `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` (Handoff Contract).

Report what the spec bounds, what it excludes, and any assumption worth rejecting now. Then end the turn with `**Next:** auto-plan, <reason>`.

## Output

- `SPEC.md`, with `canonical_spec` and `stage: frame` recorded in `current.json`. Plus `ROADMAP.md` when the user approved a decomposition.
- Halted without an approved approach, nothing is written. Report the discussion, why no approach was selected, and any deferred scope worth preserving.

## Rules

- If the user's framing shifts, reclassify and say so in plain language rather than re-routing silently.
- If the user expresses impatience, ask the two most critical unresolved questions. If they push back again, present alternatives with explicit assumptions.
- If the user tries to skip spec writing on work that spans sessions, write the smallest useful SPEC and ask them to confirm or edit it. Work below the engagement criterion left at Check Engagement and never reaches this rule.
