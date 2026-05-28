# Content Execution

Load this reference when the active slice drafts, rewrites, edits, outlines, audits, or verifies a prose artifact.

## Execution Contract

Content execution produces or changes prose, but it still follows the active slice. Do not broaden the assignment into a general writing workflow.

Before editing, identify:

- Artifact target: path or chat-native output.
- Audience: reader, context level, and intended belief or behavior change.
- Thesis: the claim to preserve.
- Voice: sample pointer or stated style constraints.
- Content Anti-Goals: named patterns to avoid.
- Channel, Source Policy, Factual Risk, and Format from the plan or SPEC.md.

If a required input is missing, stop with `NEEDS_CONTEXT` rather than filling the gap.

## Factual-Risk Gate

Never invent:

- sources, citations, quotations, links, or page titles
- metrics, dates, prices, percentages, benchmarks, or rankings
- examples framed as real events
- facts outside the allowed source policy

For high factual risk, draft only from provided or freshly verified sources. If browsing or document extraction is required but not in scope, stop and report the missing source step.

## Drafting Loop

1. Build the smallest structure that satisfies the format.
2. Draft against the thesis and audience, not against generic completeness.
3. Apply voice constraints after the factual structure is correct.
4. Run a local anti-slop pass before marking the slice complete.
5. Preserve source traceability for every non-obvious factual claim.

## Anti-Slop Pass

Scan against `.agent/.automaton/references/ANTI-SLOP.md`. Fix each hit directly, or record why it is quoted source text, required by the approved voice sample, or intentionally kept.

## Output Discipline

- Preserve existing Markdown structure unless the slice says to restructure.
- Keep comments and explanations out of the artifact unless the format requires them.
- Do not hide uncertainty with polished prose.
- If the slice is a rewrite, preserve meaning before improving style.
- If the slice is a draft, leave explicit source placeholders only when the plan allows a later sourcing pass.
