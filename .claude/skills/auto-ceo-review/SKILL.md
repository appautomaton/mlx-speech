---
name: auto-ceo-review
description: Optional product go/no-go on a framed spec. Use when product direction needs review before planning.
metadata:
  stage: frame
---

# auto-ceo-review

Optional product-direction review. Decides whether a spec is worth building before planning begins.

First action: run `node .agent/.automaton/scripts/get-context.mjs` from the project root.

## Preamble

Product bet review. Restates the objective as one crisp bet, identifies differentiation, calls out generic or mis-scoped direction. Does not design implementation or write code.

A good review names the bet in one sentence, identifies the weakest assumption, and renders a verdict in under 150 words. A bad review restates the spec.

Loading discipline: one SPEC.md read, one review paragraph, one verdict. Read project files when understanding the codebase helps ground the review. Verify that spec claims reflect what actually exists before approving or rejecting.

## Quality Gate

Before appending the product review:
- Replace strategic filler with user, action, value, and risk.
- Separate supported claims from assumptions.
- Name the strongest risk even when approving.
- Read `references/quality.md` when the review sounds like polite validation.

## Do

<GATE>

Do NOT proceed unless:
- `canonical_spec` is set and `SPEC.md` is readable.

If the spec is missing or unreadable, set verdict to `needs_clarification` and stop.
</GATE>

### Load State

Read the canonical `SPEC.md`.

### Restate the Bet

In one sentence: "We are betting that [specific user] will [specific action] because [specific reason], and the risk is [specific risk]." Read `references/bet-framing.md` for crisp-vs-vague bet examples.

### Evaluate

Choose a review posture per `references/review-modes.md`. Assess differentiation, user value, generic or mis-scoped elements, and shippability. Ground each in evidence from the spec. Read `references/product-checklist.md` for structured checks and `references/cognitive-patterns.md` for thinking patterns that surface blind spots.

### Render Verdict

Use exactly one of the four approved values.

### Verdict Values

Use strict vocabulary. No synonyms.

| Verdict | Meaning | Next Action |
|---------|---------|-------------|
| `approved` | Direction is sound. Proceed to planning. | `auto-plan` |
| `approved_with_risks` | Direction is sound but carries known risks. Document them in the review. | `auto-plan` |
| `needs_clarification` | Direction cannot be evaluated. Return to framing. | `auto-frame` or `auto-office-hours` |
| `descoped` | Direction is out of scope or low-leverage. Do not pursue. | `auto-office-hours` or stop |

### Append Review

Add a `## Review: Product` section to `SPEC.md` using the exact template in `references/review-template.md`.

### Update State

Run `node .agent/.automaton/scripts/sync-status.mjs --product-review "<verdict>"` from the project root.

### Recommend

Continue inline on a non-blocking verdict; stop and hand off on a blocking one. The verdict→skill map is in Output.

## Output

- `SPEC.md` with appended `## Review: Product` section
- `.agent/.automaton/state/current.json` updated through `sync-status.mjs` with `product_review`; `stage` is unchanged by this skill
- Handoff (verdict-mapped): `approved`/`approved_with_risks` → continue inline into `auto-plan`; `needs_clarification` → `Next: auto-frame` (or `auto-office-hours`); `descoped` → `Next: auto-office-hours`, or halt.

## Rules

- Be decisive, not theatrical. A sharp verdict is better than a long analysis.
- Do not turn the review into implementation design. Stay in product bet territory.
- Verdict vocabulary is strict. Use only the four approved values.
