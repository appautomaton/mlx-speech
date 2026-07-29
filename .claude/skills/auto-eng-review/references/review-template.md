# Review Template: Engineering

Append exactly this format to `PLAN.md`:

```markdown
## Review: Engineering

- Verdict: <approved|approved_with_risks|needs_correction>
- Strength: <one sentence>
- Concern: <one sentence, or for approved_with_risks one line per risk>
- Action: <one sentence>
- Verified: <what was checked, or "pending">
- Outside voice: <rounds run, unresolved points, log path; include only when the cross-model loop ran>
```

## Rules

- Verdict must be one of the three approved values. No synonyms.
- Strength must be exactly one sentence.
- Concern is exactly one sentence, except for `approved_with_risks`: one line per documented risk, each naming the slice it affects when known, so `auto-execute` can surface the right risk before each slice.
- Action must be a concrete next step, not a strategy.
- Verified must list what was actually checked (e.g., "data flow traced", "edge cases enumerated"), or "pending" if nothing was checked.
- Outside voice is the only conditional field: round count, unresolved points, and a pointer to `orchestration/outside-voice-log.md`, present only when the cross-model loop ran.
- Do not add fields or commentary beyond this format.
