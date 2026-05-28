# SPEC Shape

Use this before writing or refreshing `.agent/work/<change>/SPEC.md`.

SPEC.md is the reloadable contract. It must be specific enough for planning and verification, but it should link detail files under `spec/` instead of becoming a dossier.

## Core Fields

Always include:
- Bounded goal: one sentence.
- Broader intent: the larger user goal preserved or intentionally decomposed.
- Work scale and work shape.
- Selected lenses.
- Constraints and risks that change implementation.
- Required outcome in the shape the work needs: behavior, structural change, behavioral invariants, parity target or gap matrix, audit questions, migration target, coverage target, or content target.
- Acceptance criteria or traceable requirement matrix: the testable checks; do not mirror Required outcome.
- Anti-goals.

## Conditional Fields

Include only when the trigger applies:
- Linked detail files under `spec/` - trigger: constraints, risks, acceptance detail, or gap matrix is too large for inline SPEC.
- Target user or stakeholder - trigger: product, design, or content lens is selected, or INTAKE names one.
- Scope coverage decisions - trigger: intake or request includes included, deferred, anti-goal, or needs-decision items.
- Blocking questions or assumptions - trigger: present and material; omit when none.

## Shape Notes

- Refactor: structural change, behavioral invariants, blast radius, regression proof.
- Parity: reference source, gap matrix, target conformance, gap-ID verification.
- Audit: questions, evidence sources, finding schema, decision gate.
- Migration: source state, target state, rollout or rollback, verification.
- Coverage: target risk areas, expected improvement, regression proof.
- Content: audience, thesis, voice direction, content anti-goals; read `content-framing.md`.
