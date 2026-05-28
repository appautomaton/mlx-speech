# Context Loading Discipline

Internal guidelines for preserving reasoning headroom across multi-session agentic work.

## Principles

1. **Context is finite.** Every token loaded reduces headroom for reasoning. Treat context like memory, not storage.
2. **Load progressively.** Start with the smallest artifact that unlocks the next decision. Load more only when needed.
3. **Never re-read.** If you loaded a file in this session, do not read it again unless the user explicitly requests it, you know it has changed, or you are running a verification step that requires fresh evidence.
4. **Generate summaries, not transcripts.** When reporting findings, compress 500 lines of evidence into 5 lines of conclusion.
5. **Keep artifacts concrete.** Do not write context-budget fields, token-allocation notes, or percentage estimates into SPEC.md, PLAN.md, slice detail files, or evidence blocks. Artifacts record objectives, acceptance criteria, verification, dependencies, status, evidence, risks, and links.

## Progressive Loading Order

When entering any stage, load files in this order. Stop as soon as you have enough context to proceed.

```
1. .agent/.automaton/state/current.json (always, < 50 tokens)
2. SPEC.md               (if canonical_spec exists, < 1000 tokens)
3. PLAN.md               (if executing, < 1000 tokens)
4. Wiki pages            (only if referenced by spec or plan)
5. Source files          (read as needed to understand the project and produce accurate work)
```

## Artifact Language Boundary

Use this guide to decide what to load, link, summarize, or checkpoint. Do not turn the heuristic into durable artifact prose.

| Instead of... | Use... |
|---------------|--------|
| Context-size estimates in PLAN.md | `Detail: slices/slice-NNN.md` when slice instructions are too large for the plan index |
| "This is a big change" | "This requires three independently verifiable slices" |
| "Read the whole codebase" | "Load files named by the active slice; scan wider only when correctness requires it" |
| "Re-read the spec" | "The spec is already loaded. Summarize the relevant section unless this is a verification step." |

## Session Headroom

**Rule of thumb:** Keep loaded context under 60% of total window. The remaining 40% is for reasoning and response generation.

## Context Degradation Tiers

Monitor context usage and adjust behavior accordingly. These are behavioral rules, not hard limits.

| Tier | Usage | Behavior |
|------|-------|----------|
| **PEAK** | 0–30% | Full operations. Read bodies, spawn multiple agents, inline results. |
| **GOOD** | 30–50% | Normal operations. Prefer frontmatter reads, delegate aggressively. |
| **DEGRADING** | 50–70% | Economize. Frontmatter-only reads, minimal inlining, warn user about context pressure. |
| **EMERGENCY** | 70%+ | Halt new work. Checkpoint progress immediately. No new reads unless critical. |

**Warning signs before panic thresholds fire:**

- **Silent partial completion.** Agent claims task is done but implementation is incomplete.
- **Increasing vagueness.** Phrases like "appropriate handling" or "standard patterns" replace specific code.
- **Skipped steps.** Agent omits protocol steps it would normally follow.

When you see these, assume context pressure and move to a higher tier of conservation.

## Anti-Patterns

- **Broad scans.** `find . -name "*.js" | xargs cat` loads the entire codebase. Never do this.
- **Greedy wiki loading.** Loading every file in `.agent/wiki/` because "they might be useful."
- **Artifact bloat.** SPEC.md that is 800 lines long. Link detail under `spec/*.md` or move architecture rationale to DESIGN.md.
- **Re-read loops.** Reading `package.json` three times in one session because it was not held in working memory.

## No-Re-Read Rule

**Absolute rule:** Once a file is read in a session, it stays loaded. Do not read it again.

**Exceptions:**
- The user explicitly asks you to re-read it.
- You wrote to the file and need to verify the write.
- The file is known to have changed (e.g., you ran a command that mutates it).
- The current skill is an explicit verification pass, such as `auto-verify`, and fresh evidence is part of the acceptance criteria.

**If you cannot remember what a file said:** Summarize from your existing context rather than re-reading.
