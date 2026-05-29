# Librarian (Read-Only Codebase Lookup)

`automaton-librarian` is a read-only, cheap-model subagent for codebase exploration. Any stage — `auto-office-hours`, `auto-frame`, `auto-plan`, or `auto-execute` — may dispatch it to answer a bounded "where is X / how does Y connect / which files matter for Z" question without pulling wide reads into the caller's context window.

This is a one-shot lookup, not the per-slice subagent protocol. There is no review loop, and it does not make the caller an orchestrator: you ask one question, you get evidence back, and you keep every decision. The implementer and reviewer dispatch rules in `SUBAGENT-PROTOCOL.md` are separate and stay execute-only.

## When To Use

- Locating code, tracing a flow, or mapping which files a change touches would otherwise cost many wide reads in the parent session.
- You need facts to frame, scope, or plan — not a recommendation about what to build.

Do not dispatch it for a lookup you can settle in one or two reads yourself; the dispatch overhead only pays off when exploration would otherwise bloat your context.

## What It Guarantees (And Does Not)

- Structurally read-only on every host (Claude `tools: Read, Grep, Glob`; Codex `sandbox_mode=read-only`; OpenCode `edit/bash/task: deny`). It cannot mutate files.
- Cheap where a light model is configured for it (Claude: `haiku`; Codex: low reasoning effort; other hosts only if the deployment pins one). If no light model is available it still works, but the cheap-call guarantee no longer holds — treat that as a known limit, not a silent fallback to the parent model.
- It returns evidence, never decisions. Any scope or approach opinion arrives under `UNCERTAINTY` for the caller to judge.

## Dispatch

Dispatch the host-native agent named `automaton-librarian` with your host's subagent mechanism (see `HOST-TOOLS.md` for the exact call on this host). Pass one bounded question:

```text
<question>
the specific where / how / which-files question
</question>

<scope-hint>
optional: directory, subsystem, or files to start from
</scope-hint>
```

It replies with `STATUS`, `ANSWER`, `FILES` (with `path:line` anchors), `RELATIONSHIPS`, `UNCERTAINTY`, and `NEXT_READS`. Read its summary and act on it; do not paste its full reply into a durable artifact.
