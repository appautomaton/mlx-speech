# Content Framing

Load this reference when the change involves content creation: writing, articles, briefs, decks, newsletters, documentation, or any deliverable where prose quality matters.

## Content-Aware SPEC.md Fields

When framing a content-oriented change, add these fields to SPEC.md alongside the standard bounded goal, lenses, constraints, and anti-goals:

### Audience

One sentence: who reads this, what they already know, and what belief or behavior the content should change. Name the misconception, not the field. This is the level of specificity to hit:

"Senior engineers who know distributed systems but assume event sourcing is only for CQRS. This piece argues it's a general-purpose audit pattern."

### Thesis

One falsifiable or debatable claim the piece makes. Not a topic, not a summary. A position someone could argue with:

"Feature flags cost more in maintenance debt than they save in deployment safety, and most teams should delete theirs."

### Voice

Either a pointer to a voice sample (file path or inline excerpt) or a description of the target voice concrete enough to write from: sentence rhythm, formality level, use of first person, punctuation habits.

"Short sentences, contractions, first person. Reads like a senior engineer explaining to a peer, not lecturing. No hedging. State positions directly."

### Content Anti-Goals

What the content must not sound like, as structural patterns rather than abstract qualities. "High quality" and "engaging" set no boundary. These do:

- No significance inflation: stakes the evidence does not support.
- No em-dashes as connective tissue, and no three-part list whose third item is there for rhythm.
- No sycophantic framing or signposting before the content starts.
- No promotional adjectives. Describe the thing rather than praising it.

## Anti-Slop Checklist

Before finalizing a content-oriented SPEC.md, scan the spec itself against `.agent/.automaton/references/ANTI-SLOP.md`. A spec that tells the implementer to avoid slop but models sloppy prose undermines the direction.

## Lens Interaction

The content lens rule lives here; skills and stage references point instead of restating it.

- Content-only change (article, blog post, newsletter): lenses are `product` + `content`. Add `design` when the deliverable has a visual surface (deck, styled docs page).
- Content inside a feature (onboarding copy, error messages, docs): lenses are `product` + `engineering` + `content`.
- The content lens never triggers `security` or `runtime` unless the content touches sensitive data or is generated at runtime.

## Deferred Dimensions

Capture these when the user already supplied them or when they materially affect scope. Otherwise leave them for planning as explicit assumptions or blocking questions:

- **Channel:** where the content will be published (blog, docs site, newsletter, social).
- **Source policy:** what can be cited, linked, or assumed as common knowledge.
- **Factual risk:** how much fact-checking the content requires (opinion piece vs. technical reference).
- **Format:** structural template (listicle, narrative, tutorial, reference doc).
