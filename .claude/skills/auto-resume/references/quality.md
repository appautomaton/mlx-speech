# Resume Quality

Load this reference only before producing the recovery summary or next-action recommendation.

## Resume Anti-Patterns

- Invented continuity: filling missing state from memory or likely intent.
- Narrative recap: long story of prior work instead of current durable facts.
- Stale-pointer concealment: glossing over missing canonical artifacts.
- Broad context loading: reading wiki or future-stage files before the current stage requires them.
- Generic next step: recommending a skill without naming why.

## Better Shape

- Prefer `current.json` and canonical artifacts over conversation memory.
- Report missing or stale pointers plainly.
- Summarize only what changes the next action.
- Name the exact next skill and the blocker it resolves.

## Prose Hygiene

Recovery summaries attract narrative padding. The summary should orient, not narrate.

Scan for:
- "Previously, we had been working on...": state the current artifact, not the journey
- "significant progress was made": name the last completed slice
- "the project is in good shape": name what is done and what is blocked
- Em dashes and -ing phrases in the summary
- Any sentence that would still sound valid after changing the change name

Before: "Significant progress has been made on the authentication overhaul — previously we'd been exploring various approaches, and the team has been working diligently on the implementation."
After: "Active change: auth-overhaul. Stage: execute. Slice 2 of 4 complete. Blocked: PLAN.md references deleted migration file."

## Final Check

If the summary would still sound valid after changing the active change name, it is too generic.
