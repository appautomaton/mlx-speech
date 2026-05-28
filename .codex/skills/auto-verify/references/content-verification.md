# Content Verification

Load this reference when verifying a slice that creates, rewrites, edits, outlines, or audits prose.

## Verification Contract

Evaluate each check as PASS, FAIL, or PARTIAL with evidence. Do not collapse content quality into "reads well".

| Check | Evidence requirement |
| --- | --- |
| Audience | Name the intended reader and cite where the artifact addresses their knowledge, belief, or action need. |
| Thesis | Quote or summarize the core claim and confirm each major section supports it. |
| Voice | Compare the artifact to the voice sample or stated voice direction using sentence rhythm, formality, point of view, and vocabulary. |
| Content Anti-Goals | List the named anti-goals and show whether the artifact violates any of them. |
| Channel | Confirm the artifact fits the publication surface, such as docs, blog, newsletter, deck, proposal, README, or internal brief. |
| Source Policy | Verify every citation, link, quotation, and external fact is allowed by the plan. |
| Factual Risk | Classify the risk level and confirm the evidence standard was met. |
| Format | Confirm the structure matches the required format and artifact target. |

## Anti-Slop Pattern Scan

Scan the final artifact against `.agent/.automaton/references/ANTI-SLOP.md`. Each hit is FAIL unless it is quoted source text, required by the approved voice sample, or intentionally justified.

## Source And Fact Checks

- For provided-source-only work, every factual claim must trace to the provided material or be removed.
- For high-risk content, current or technical claims need explicit source evidence.
- For rewrite work, verify no new claims were introduced without permission.
- For citations, confirm the target exists when the verification environment permits it.

## Report Shape

Use the standard verification report, then add a content subsection:

```
Content checks: PASS / FAIL / PARTIAL
Audience: [evidence]
Thesis: [evidence]
Source policy: [evidence]
Anti-slop scan: [hits or none]
```
