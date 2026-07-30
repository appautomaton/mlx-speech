# Content Framing

Load this when the change involves content creation: writing, articles, briefs, decks, newsletters, documentation, or any deliverable where prose quality matters. It carries both the content-mode diagnostic and the SPEC fields that diagnostic fills.

Content is a peer mode alongside Startup and Builder, not an overlay on them. When the mode is Content, these questions replace the Startup or Builder routing.

## Content-Mode Diagnostic And SPEC Fields

Four required-core fields. Ask one at a time, push until the answer reaches the stated bar, then write it into SPEC.md. The definition and the bar are the same thing: an answer that does not meet the bar has not filled the field.

### Audience

Who reads this, what they already know, and what belief or behavior the content should change. Name the misconception, not the field. A role with no context level, or "anyone interested in X", has not answered.

"Senior engineers who know distributed systems but assume event sourcing is only for CQRS. This piece argues it's a general-purpose audit pattern."

### Thesis

One falsifiable or debatable claim the piece makes. Not a topic, not a summary. "An overview of X" is a topic. A thesis takes a position someone could argue with.

"Feature flags cost more in maintenance debt than they save in deployment safety, and most teams should delete theirs."

### Voice

A pointer to a voice sample (file path or inline excerpt), or a description concrete enough to write from: sentence rhythm, formality level, use of first person, punctuation habits. "Professional tone" is the absence of a voice, not a voice.

"Short sentences, contractions, first person. Reads like a senior engineer explaining to a peer, not lecturing. No hedging. State positions directly."

When a sample exists, analyze sentence length, word choice level, paragraph openers, punctuation habits, and transitions, then match them in all downstream content work. When none exists, ask how they would explain this to a colleague and calibrate from that answer.

### Content Anti-Goals

What the content must not sound like, as structural patterns rather than abstract qualities. "High quality" and "engaging" set no boundary. Push for a named negative example with a reason. These do set a boundary:

- No significance inflation: stakes the evidence does not support.
- No em-dashes as connective tissue, and no three-part list whose third item is there for rhythm.
- No sycophantic framing or signposting before the content starts.
- No promotional adjectives. Describe the thing rather than praising it.

Name the pattern from `.agent/.automaton/references/ANTI-SLOP.md` rather than a vague goal, then carry it into this field.

## Lens Interaction

The content lens rule lives here; skills and stage references point instead of restating it.

- Content-only change (article, blog post, newsletter): lenses are `product` + `content`. Add `design` when the deliverable has a visual surface (deck, styled docs page).
- Content inside a feature (onboarding copy, error messages, docs): lenses are `product` + `engineering` + `content`.
- The content lens never triggers `security` or `runtime` unless the content touches sensitive data or is generated at runtime.

## Deferred Dimensions

Capture these when the user already supplied them or when they materially affect scope. Otherwise leave them for planning as explicit assumptions or blocking questions:

- **Channel:** where the content will be published (blog, docs site, newsletter, social).
- **Source Policy:** what can be cited, linked, or assumed as common knowledge.
- **Factual Risk:** how much fact-checking the content requires (opinion piece versus technical reference).
- **Format:** structural template (listicle, narrative, tutorial, reference doc).

## Anti-Slop Checklist

Before finalizing a content-oriented SPEC.md, scan the spec itself against `.agent/.automaton/references/ANTI-SLOP.md`. A spec that tells the implementer to avoid slop but models sloppy prose undermines the direction.
