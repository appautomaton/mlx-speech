# Content Planning

Load this reference when `SPEC.md` has content fields or the change is about writing, articles, briefs, decks, newsletters, documentation, proposals, or rewrite passes.

## Inputs From Framing

Carry these framing fields into every content slice:

| Field | Planning use |
| --- | --- |
| Audience | Names the reader and the behavior or belief the content must change. |
| Thesis | Becomes the invariant claim every slice must preserve. |
| Voice | Constrains tone, sentence rhythm, formality, and point of view. |
| Content Anti-Goals | Defines what the artifact must not sound like or do. |

If a required-core field (Audience, Thesis, Voice, Content Anti-Goals) is missing from SPEC.md, that is a framing gap: stop and ask, or recommend `auto-frame`. Never fill it with a planning assumption. Assumptions are for the deferred dimensions below, and only when execution does not hinge on the answer.

## Deferred Dimensions

Add these when they affect execution or verification:

- **Channel:** where the artifact will live, such as docs site, blog, newsletter, deck, proposal, README, or internal brief.
- **Source Policy:** what may be cited, quoted, linked, or treated as common knowledge.
- **Factual Risk:** low for opinion or positioning, medium for explainers, high for technical claims, legal/medical/financial claims, pricing, current facts, or sourced comparisons.
- **Format:** the required structure, such as narrative article, tutorial, reference doc, memo, deck outline, changelog, or rewrite diff.

## Slice Shape

Content slices should name:

1. Artifact target: exact output path or "chat-native output only".
2. Source inputs: files, URLs, pasted material, or "none allowed".
3. Drafting unit: outline, section, full draft, rewrite pass, source audit, or final polish.
4. Verification command: a concrete check, not "review quality".

Prefer Markdown-first artifacts unless SPEC.md names a different format.

## Source And Factual Gates

- If factual risk is high, add a slice that gathers or validates sources before drafting.
- If source policy is "provided sources only", execution must not add outside facts.
- If sources are required but missing, the plan must stop at a blocking question.
- If citations are needed, verification must check that every factual claim traces to an allowed source.

## Example Slices

### Slice: Draft Technical Blog Outline

- **Artifact target:** `.agent/work/<change>/outline.md`
- **Inputs:** SPEC.md audience, thesis, voice, anti-goals. Source notes in `research.md`
- **Constraints:** channel = blog, format = outline, factual risk = medium
- **Verification:** check headings support the thesis, name the audience, and include source placeholders for factual claims

### Slice: Rewrite Documentation Page

- **Artifact target:** `docs/foo.md`
- **Inputs:** existing doc and SPEC.md voice
- **Constraints:** channel = docs site, source policy = existing repo only, format = reference doc
- **Verification:** run markdown/lint tests when available and inspect for audience fit, preserved facts, and anti-goal violations
