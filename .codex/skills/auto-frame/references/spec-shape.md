# SPEC Shape

Use this before writing or refreshing `.agent/work/<change>/SPEC.md`.

`SPEC.md` is the reloadable contract. It must be specific enough for planning and verification, and it should link detail files under `spec/` instead of becoming a dossier. It records decisions, not the conversation that produced them.

## Core Fields

Always include:
- **Bet:** a one line `**Bet:** <wager>` opening the spec. It names what the change bets in plain language, so the reader can reject the premise before reading the mechanics.
- Bounded goal: one sentence, in the user's final refined language.
- Work scale and work shape.
- Selected lenses.
- Constraints and risks that change implementation.
- Required outcome in the shape the work needs: behavior, structural change, behavioral invariants, parity target or gap matrix, audit questions, migration target, coverage target, or content target.
- Acceptance criteria or traceable requirement matrix: the testable checks. Do not mirror Required outcome.
- Anti-goals.

## Conditional Fields

Include only when the trigger applies:
- Linked detail files under `spec/` - trigger: constraints, risks, acceptance detail, or gap matrix is too large for inline SPEC.
- Target user or stakeholder - trigger: product, design, or content lens is selected.
- Scope coverage: included, deferred, anti-goal, and resolved needs-decision items - trigger: the request carried more than one material ask. Omit empty groups.
- Broader intent: the larger user goal preserved or intentionally decomposed - trigger: it differs from the bounded goal.
- Scope preservation: whether this preserves the user's full stated intent or intentionally decomposes it - trigger: anything was deferred or narrowed.
- Approved approach: the chosen approach, the evidence supporting it, and what that evidence does not prove - trigger: alternatives were presented.
- Rejected or deferred framings, with reasons - trigger: a direction was ruled out. Do not carry the full alternatives analysis.
- Mode context - trigger: the mode changes framing. Startup: demand, status quo or workaround, target user or wedge. Builder: core delight, novelty, the "whoa" factor. Content: reader, thesis, voice, content anti-goals.
- Blocking questions or assumptions - trigger: present and material.
- `Supersedes:` in the header - trigger: a prior spec exists for this change.

## Shape Notes

Include the section that changes framing or verification. Omit for plain feature shape when mode context is enough. Do not use delight language for parity, audit, refactor, migration, or coverage work.

- Refactor: structural goal, behavioral invariants, blast radius, regression proof.
- Parity: reference source, gap landscape, what "closed" means, gap-ID verification.
- Audit: what the audit must answer, evidence sources, finding schema, the decision that depends on findings.
- Migration: source state, target state, compatibility constraints, rollout or rollback, verification.
- Coverage: target risk areas, what is undertested, expected improvement, regression proof.
- Content: audience, thesis, voice, content anti-goals. Read `content-framing.md`.

Save to `.agent/work/<change>/SPEC.md`, never to a host-specific path. A `SPEC.md` without `canonical_spec` in `current.json` means framing is still in progress.
