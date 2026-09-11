# MLX Speech audio studio website

Audience: developers choosing a local speech model for Apple Silicon.
Thesis: hear a real output, find a suitable model, and run a complete example.
Voice: concise, specific, welcoming. Name models, tasks, and requirements.
Content anti-goals: invented benchmarks, simulated live inference, and generic AI claims.

The user requested a distinctive redesign and selected the audio-studio direction:
waveform visuals, strong typography, and model demos. The existing page passes
Lighthouse's technical search checks but has contrast and heading defects.

Acceptance criteria:

- Replace the current card-heavy presentation with an audio-studio composition,
  pronounced typography, a real waveform, and a working prerecorded demo player.
- Give this project its own visual identity. Provide three curated palettes,
  each with a complete light and dark treatment. Select a different palette on
  reload, following the system theme initially and remembering an explicit
  visitor theme choice. Preserve readable contrast across all six combinations.
- Use the repository's documented VibeVoice output as the initial demo. Identify
  its model and prerecorded nature, preserve provenance, and provide a transcript.
- Retain all 15 model variants, their correct selectors, guides, and weights.
  Offer useful task filters with all models available when scripts are disabled.
- Provide complete TTS, cloning, and transcription examples, explicit Apple
  Silicon/Python prerequisites, and the FireRed source-install requirement.
- Preserve the production URL, valid canonical/social/structured metadata,
  working catalog links, and static crawlable content. Update the sitemap date.
- Validate desktop, mobile, keyboard operation, reduced motion, audio playback,
  no-script access, internal links, and absence of horizontal overflow.
- Keep Lighthouse SEO at 100 and resolve the identified accessibility defects.
  Run the repository's complete unit suite before reporting completion.

Scope: `site/`, the existing catalog regression tests in
`tests/unit/test_site_catalog.py`, and the planning records. Keep catalog tests
focused on model facts, crawlable links, and accessible names across the markup change.
No runtime/model implementation changes,
new application framework, or dependency additions. Account audit data remains
outside the public repository. The separately authorized Hugging Face URL changes
have been published and verified across 14 model cards.
