# Implementation Alternatives

Use this only when PLAN.md lacks an approach rationale, the user asks for alternatives, or the review verdict depends on comparing safer execution paths. Keep it in chat unless the plan needs correction.

For each approach:

```
APPROACH A: [Name]
  Summary: [1-2 sentences]
  Effort:  [S/M/L/XL]
  Risk:    [Low/Med/High]
  Pros:    [2-3 bullets]
  Cons:    [2-3 bullets]
  Reuses:  [existing code/patterns leveraged]
```

Rules:
- At least 2 approaches when alternatives are needed. 3 only when the decision is genuinely high-leverage.
- One approach must be the "minimal viable" (fewest files, smallest diff).
- One approach must be the "ideal architecture" (best long-term trajectory).
- These two approaches have equal weight. Do not default to minimal just because it is smaller.
- If only one approach exists, explain concretely why alternatives were eliminated.
- Do not write alternatives into PLAN.md unless the verdict is `needs_correction` and the alternatives are the correction path.

Recommend one with a one-line reason mapped to engineering preferences.
