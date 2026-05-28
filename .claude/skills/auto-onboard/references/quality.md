# Onboard Quality

Load this reference only before writing or refreshing steering artifacts.

## Anti-Patterns

- Uncited repo claims: architecture, commands, or ownership stated without file evidence.
- Repo-map bloat: exhaustive inventory instead of bounded project truth.
- Question parking: adding speculative `Open Questions` that do not block steering.
- Routing chatter: adding recommended next skill or confidence verdicts to REPO-MAP.md.
- Template gravity: filling every scaffold section instead of deleting sections with no durable evidence.
- Guessed topology: treating familiar framework names as proof without checking entrypoints.
- Stale-state overwrite: replacing useful steering without naming what changed.
- Scanning past sufficiency: continuing broad reads after the repo shape is clear.

## Better Shape

- Lead each steering artifact with the current truth, then cite paths.
- Mark uncertain claims as unknown or inferred.
- Keep uncertainty only when it changes steering or the immediate next action.
- Delete empty template sections. Do not leave placeholder-shaped prose in finished artifacts.
- Keep commands to those observed in manifests, scripts, docs, or working checks.
- Stop when the next action is clear enough for `auto-frame` or `auto-resume`.

## Prose Hygiene

Steering artifacts attract promotional language and uncited claims. Every statement should cite a file path or mark itself as inferred.

Scan for:
- "robust architecture", "well-structured codebase": name the pattern and where it lives
- "comprehensive test suite": name the test runner and count
- "modern technology stack": name the runtime, framework, and version
- Promotional adjectives about the repo's quality
- Speculative questions such as "is this intended to be user-facing?" when the artifact can simply state what was observed
- `Import Verdict`, `steering confidence`, or `recommended next skill` sections in REPO-MAP.md
- headings copied from templates that have only weak, inferred, or redundant content
- Any claim without a file path citation

Before: "This is a well-structured, modern codebase with a robust architecture and comprehensive test coverage, reflecting best practices in full-stack development."
After: "Node.js 20 + Express 4 API (src/server.js). React 18 SPA (client/src/). 47 test files under test/ using Vitest. No CI config found (inferred: not yet configured)."

## Final Check

If a fresh agent could not tell which facts were observed versus inferred, revise the artifact.
