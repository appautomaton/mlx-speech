---
name: auto-frame
description: Bound and de-risk a request into SPEC.md. Use when the objective is clear but scope needs constraining.
metadata:
  stage: frame
---

# auto-frame

Framing controller. Bounds and de-risks a request into a single `SPEC.md`.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

auto-frame produces the canonical artifact: `SPEC.md` when the request is frameable. SPEC.md is mandatory for frame completion; no file means no completed frame. It does not write code, create PLAN.md, or proceed to planning without a written spec. If one focused framing question cannot make the request frameable, continue into `auto-office-hours` rather than writing a weak SPEC.

Loading discipline: hold the INTAKE when present, the objective, constraints, risks, and source evidence needed to keep the spec real. Avoid exhaustive tree walks. When locating code or tracing a flow would otherwise pull wide reads into context, you may dispatch the read-only `automaton-librarian` for a one-shot lookup (see `.agent/.automaton/references/LIBRARIAN.md`); it returns evidence, you keep the decision.

Artifact discipline: `SPEC.md` is the reloadable contract, not the whole dossier. Keep it compact enough to re-read; for large coherent work, summarize the contract and link normative detail under `spec/*.md`. One coherent outcome remains one spec even when it needs progressive disclosure.

## Quality Gate

Before finalizing `SPEC.md`:
- Make the objective observable.
- Preserve the user's broader intent; do not silently narrow scope.
- Move implementation detail out unless it constrains scope.
- Mark uncertain claims as assumptions.
- Read `references/quality.md` when the spec feels broad, padded, or hard to verify.

## Do

### Restate

Read `.agent/work/<active_change>/INTAKE.md` if it exists. `INTAKE.md` is preferred context, not a prerequisite for framing. If no intake exists, frame from the current request, conversation context, and repo evidence. Do not send the user back to office-hours solely because `INTAKE.md` is missing.

Adopt settled office-hours context: work scale, work shape, Broader intent, target user or stakeholder, scope coverage, rejected framings, and anti-goals. Do not re-ask settled context or reintroduce rejected directions.

State the goal in one sentence. If you cannot, ask one clarifying question. If the request still needs objective discovery or multiple material decisions before any useful SPEC can be written, continue into `auto-office-hours`'s diagnostic and intake flow in the same session.

If your SPEC would be narrower than the user's stated goal or office-hours broader intent, widen the SPEC, ask for confirmation, or record the deferred scope as a `Deferred / Not in scope` note inside this change's SPEC. Do not create `ROADMAP.md` phases from a narrowed SPEC; phased decomposition belongs to `auto-office-hours` after the user approves it. Silent narrowing is a framing failure.

### Coverage Check

If `INTAKE.md` or conversation context includes scope coverage, compare the intended SPEC against each item before writing:
- Included items must appear in the bounded goal, required outcome, constraints, risks, or acceptance criteria.
- Deferred items must stay deferred with a reason in a SPEC deferred-scope note.
- Anti-goals must appear in SPEC anti-goals.
- Needs-decision items require one focused question or 2–3 concrete options before SPEC unless the user explicitly accepts an assumption.

If no formal scope coverage exists but the request has multiple material asks, perspectives, constraints, or worries, build the lightweight check from available context. Do not drop a material item silently.

### Surface

List only constraints, unknowns, and risks that change implementation or verification. Keep decision-critical material in `SPEC.md`; link larger coherent detail under `spec/constraints.md`, `spec/risks.md`, `spec/gap-matrix.md`, or similar. If constraints point to unrelated outcomes, ask which outcome to frame first.

### Select Lenses

Choose the minimum useful lenses from `product`, `engineering`, `design`, `security`, `runtime`. Default to `product` + `engineering` unless the request says otherwise. Read `references/lens-selection.md` when selection is not obvious.

If the change involves content creation - writing, article, brief, deck, blog post, newsletter, documentation, or similar prose - add the content lens and read `references/content-framing.md`.

### Interview

<INTERVIEW>

Skip this when the goal, scope, and lenses are clear.

Ask only questions that materially change the SPEC. Prefer 2-4 concrete options with one-line reasons when choices are known. Use the host question tool when available; otherwise present the same options inline. Resolve `Needs decision` items before writing unless the user explicitly accepts an assumption.
</INTERVIEW>

### Continue To Office-Hours When Not Frameable

Use this only when one focused framing question is not enough. Continue into `auto-office-hours` in the same session when the request lacks a problem, stakeholder, desired outcome, content audience/thesis, first independent outcome, or direction choice. Recommend `auto-office-hours` only when continuation is blocked by context pressure, host limits, or the user's choice to pause.

When this happens, follow `auto-office-hours`'s contract: classify mode, scale, and shape; run the minimum diagnostic; present approaches; wait for approval before writing `INTAKE.md`. Do not write SPEC.md until an approach is approved and frame-ready.

### Write SPEC.md

If a `SPEC.md` already exists for this change, read it and preserve all `## Review:` sections.

<GATE>

Do NOT proceed past this step without writing `SPEC.md` to `.agent/work/<change>/SPEC.md`.

Do NOT write `SPEC.md` while a needs-decision item would change scope, approach, or verification unless the user answers it or explicitly accepts an assumption.

Read `references/spec-shape.md` and write the SPEC with its **core** fields and **conditional** fields. Conditional fields appear only when their named trigger applies.

Apply `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md` while writing: no mirror sections, index over transcript, append-replace repeated sections, and replace prior `## Review:` sections on refresh.
</GATE>

### Update State

If `active_change` is `bootstrap` or does not match the current objective, derive a new slug: `YYYY-MM-DD-<kebab-case-objective>` using today's date. Use that slug before writing SPEC.md.

After writing SPEC.md, run `node .agent/.automaton/scripts/sync-status.mjs --active-change "<change>" --canonical-spec ".agent/work/<change>/SPEC.md" --stage frame` from the project root. Use `--stage plan` only when the user approved direct plan handoff and no review is needed.

## Output

- Frameable path: **SPEC.md** written to `.agent/work/<change>/SPEC.md`; `canonical_spec` and frame state recorded through `sync-status.mjs`.
- Not-frameable path: continue into `auto-office-hours`'s contract and do not report framing complete until an approved intake exists and SPEC.md can be written.
- Handoff: after SPEC.md, continue inline into `auto-plan` when no review is needed and context is healthy; for the optional `auto-ceo-review`, stop with `Next: auto-ceo-review` rather than auto-running it. If not frameable, continue inline into `auto-office-hours` with the concrete blocker.

## Rules

- **SPEC.md is mandatory for frame completion.**
- **INTAKE.md is optional.** Use it when present, but a clear current request can be framed without it.
- If the user tries to skip spec writing, write the smallest useful SPEC and ask them to confirm or edit it.
- Ask at most three framing questions, or at most five for capability-sized goals without office-hours context; more than that belongs in office-hours.
- Match SPEC shape to the work shape; do not force every SPEC into a feature template.
- Preserve review sections on refresh.
- Keep notes operational: the SPEC is a contract, not an essay.
