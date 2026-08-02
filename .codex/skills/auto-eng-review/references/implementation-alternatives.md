# Implementation Alternatives

Use this only when PLAN.md lacks an approach rationale, the user asks for alternatives, or the review verdict depends on comparing safer execution paths. Keep it in chat unless the plan needs correction.

auto-frame keeps a sibling alternatives format for pre-spec scoping. The shared skeleton is deliberate and the rules differ on purpose: a scoping choice and an execution path answer different questions.

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
- One approach must be the **direct path**: the fewest new abstractions and dependencies needed to execute the plan safely.
- One approach must be the **ideal architecture**: the structure that ages best.
- The two carry equal weight. They differ in structural commitment, never in the quality of what gets built. A diff is smaller because the design needed less, never because the work was done to a lower standard.
- If only one approach exists, explain concretely why alternatives were eliminated.
- Do not write alternatives into PLAN.md unless the verdict is `needs_correction` and the alternatives are the correction path.

Recommend one with a one-line reason mapped to engineering preferences.
