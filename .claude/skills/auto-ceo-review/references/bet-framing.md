# Product Bet Framing

## Crisp Bets

A crisp product bet names a specific user, a specific action, a specific reason, and a specific risk.

- GOOD: "We are betting that data analysts at mid-market SaaS companies will pay $50/month for automated anomaly detection in their warehouse because they currently spend 4 hours/week writing SQL queries that miss edge cases."
- GOOD: "We are betting that mobile developers will integrate our SDK because push notification delivery is currently unreliable and they lose 15% of re-engagement traffic."

## Vague Bets

- BAD: "We are building an AI platform for enterprises." (No specific user, no specific action, no specific reason.)
- BAD: "We are making onboarding better." ("Better" is not measurable. No baseline, no target.)
- BAD: "We are betting that the market will grow." (Market growth is not a product bet.)

## Reframing Exercise

If the bet is vague, reframe using this structure:

> We are betting that [specific user] will [specific action] because [specific reason], and the risk is [specific risk].

If any bracket is missing, the bet is incomplete.

## The 10x Check

For SCOPE EXPANSION and SELECTIVE EXPANSION modes, ask:

> What is the version that is 10x more ambitious and delivers 10x more value for 2x the effort? Describe it concretely.

This is not about adding features. It is about reimagining the user experience. Start from what the user feels, not from architecture.

## Platonic Ideal

> If the best engineer in the world had unlimited time and perfect taste, what would this system look like? What would the user feel when using it?

Describe the felt experience first. Close with concrete effort and impact.

## Dream State Mapping

Describe the ideal end state 12 months from now. Does this plan move toward that state or away from it?

```
  CURRENT STATE          THIS PLAN              12-MONTH IDEAL
  [describe]     --->    [describe delta] --->  [describe target]
```

## Temporal Interrogation

Think ahead to implementation: what decisions should be resolved NOW in the plan?

```
  HOUR 1 (foundations):     What does the implementer need to know?
  HOUR 2-3 (core logic):    What ambiguities will they hit?
  HOUR 4-5 (integration):   What will surprise them?
  HOUR 6+ (polish/tests):   What will they wish they'd planned for?
```

Surface these as questions for the user NOW, not as "figure it out later."

## Expansion Framing Pattern

FLAT (avoid): "Add real-time notifications. Users would see workflow results faster, latency drops from ~30s polling to <500ms push. Effort: ~1 hour."

EXPANSIVE (aim for): "Imagine the moment a workflow finishes. The user sees the result instantly, no tab-switching, no polling, no 'did it actually work?' anxiety. Real-time feedback turns a tool they check into a tool that talks to them. Concrete shape: WebSocket channel + optimistic UI + desktop notification fallback. Effort: human ~2 days / CC ~1 hour. Makes the product feel 10x more alive."

Both are outcome-framed. Only one makes the user feel the cathedral. Lead with the felt experience, close with concrete effort and impact.

## Completeness Is Cheap

AI coding compresses implementation time 10-100x. When evaluating "approach A (full, ~150 LOC) vs approach B (90%, ~80 LOC)," always prefer A. The 70-line delta costs seconds with AI assistance. "Ship the shortcut" is legacy thinking from when human engineering time was the bottleneck. Boil the lake.
