# Content Intake

Load this reference when the user's request is about content creation: writing, articles, briefs, decks, newsletters, blog posts, documentation, or any deliverable where prose quality matters.

## Content-Mode Diagnostic

Ask these one at a time, after mode detection (Startup or Builder). These replace the mode-specific questions only when the user's goal is a content deliverable, not a product or feature.

### Q1: Audience

> "Who reads this? Not 'developers' or 'executives,' name the person. What do they already know? What do they believe that you need to change or reinforce?"

Push until you hear: a role with context level, an existing belief or knowledge gap, and what the reader should do after reading.

Red flags: "General audience." "Anyone interested in X." If they can't name the reader, the content has no perspective.

### Q2: Thesis

> "Say the one thing this piece argues or proves. Not the topic, the claim. If the reader remembers one sentence next week, what is it?"

Push until you hear: a falsifiable or debatable statement, not a topic label.

Red flags: "An overview of X." "Everything you need to know about Y." Those are topics, not theses. A thesis takes a position.

### Q3: Anti-Goals

> "What should this piece never sound like? Name a specific example, an article, a tone, a style, that would make you cringe if your piece resembled it."

Push until you hear: a concrete anti-example with a reason (e.g., "not like a ChatGPT blog post, no em-dash lists, no 'let's dive in,' no rule-of-three conclusions").

Red flags: "Just make it good." "Professional but approachable." These are non-answers. Press for a named negative example.

### Q4: Voice Sample

> "Show me something you've written before that sounds like you, an email, a doc, a tweet thread, anything. If you don't have one, describe how you'd explain this to a colleague over coffee."

If a sample is provided: analyze sentence length patterns, word choice level, paragraph openers, punctuation habits, recurring phrases, and transition style. Match these in all downstream content work.

If no sample: use the coffee-explanation answer to calibrate formality, vocabulary, and rhythm. Default to direct, varied, opinionated prose, not neutral reporting.

Red flags: "Use a professional tone." That's not a voice, it's an absence of one. Push for specifics: short or long sentences? Contractions or formal? First person or third?

## Routing

After content intake, the diagnostic output feeds into:
- `auto-frame` with a content lens: audience, thesis, voice, and anti-goals become SPEC.md fields.
- Future Pass 2 skills: channel, source policy, factual risk, and format are not captured here.

## Anti-Slop Calibration

When discussing content quality, use concrete pattern names from `.agent/.automaton/references/ANTI-SLOP.md` rather than vague goals like "high quality," "engaging," or "natural." Name the specific tell the user wants to avoid, then carry it into Content Anti-Goals.
