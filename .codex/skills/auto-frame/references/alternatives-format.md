# Alternatives Format

Present exactly 2–3 approaches using this format. For bug, feature, and capability work, one must be minimal viable and one must be ideal architecture. One can be creative/lateral. For refactor, parity, audit, migration, or coverage work, differentiate by blast radius, traceability, evidence depth, rollout risk, or verification strength instead. For roadmap-sized work, present decomposition strategies or first-spec candidates.

auto-eng-review keeps a sibling implementation-alternatives format for execution paths. The shared skeleton is deliberate and the rules differ on purpose: this file differentiates scoping choices before a spec exists, that one compares safer execution paths for an already-approved plan.

State your recommendation and its one-sentence why first, then the approaches. Present the choice through the host question tool per `.agent/.automaton/references/FRAMEWORK.md` (Asking The User), naming your recommended approach as the first option; the tool renders the options, so keep each approach compact.

```
## Approach A: [Name]

**Summary:** [1–2 sentences]
**Effort:** [S/M/L/XL]
**Risk:** [Low/Med/High]
**Pros:** [2 bullets]
**Cons:** [2 bullets]
**Reuses:** [existing code/patterns leveraged, or "none"; optional]

## Approach B: [Name]

...

## Approach C: [Name] (optional)

...
```

## Rules

- At least 2 approaches required. 3 preferred for non-trivial designs.
- For bug, feature, and capability work, one must be the **minimal viable**: the smallest version of the user's stated goal that delivers the core value. This is the leanest path to the user's goal, not a different, smaller goal.
- For the same shapes, one must be the **ideal architecture**: best long-term trajectory, most elegant.
- For other shapes, the differentiation axis in the header paragraph replaces the minimal-viable and ideal-architecture pair.
- One can be **creative/lateral**: unexpected approach, different framing of the problem.
- Alternatives must vary the approach to the user's goal, not vary the goal itself. A capability-sized goal should produce capability-sized alternatives, not three ways to shrink the goal to feature-size.
- The recommendation leads: state it and its one-sentence why before the approaches, never after.
- Do NOT proceed until the user explicitly approves an approach or chooses a different one.
