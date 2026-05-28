# Host Tools

Host: `claude`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Automaton Subagents

`auto-execute` dispatches three host-native subagents that `installHost()` wrote into this host's agent directory:

- `automaton-implementer` — Implements exactly one approved Automaton plan slice from coordinator-provided context and returns evidence.
- `automaton-spec-reviewer` — Reviews spec compliance for one approved Automaton plan slice. Verdict only; no edits.
- `automaton-quality-reviewer` — Reviews maintainability and regression risk for one approved Automaton plan slice. Verdict only; no edits.

Their static role bodies are baked into the host agent files. The coordinator fills per-call slots in `auto-execute/references/*-prompt.md` (slice, constraints, acceptance criteria, implementation summary) and hands the packet to the named agent.

## Dispatch

- availability: available
- dispatch: Use the Agent tool with `subagent_type` set to `automaton-implementer`, `automaton-spec-reviewer`, or `automaton-quality-reviewer`. Pass the per-call dispatch packet (slice, constraints, acceptance criteria, implementation summary) as the prompt; the static role body is already installed under `.claude/agents/` and Claude Code blocks the Agent tool inside subagent sessions so recursion is structurally impossible.
- wait: Agent tool calls return their result to the coordinator when complete; no separate wait command is needed.
- cleanup: No explicit close step is needed after an Agent result is returned.
- tracking: Use TodoWrite for session-local progress tracking when useful.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Dispatch only by named agent (`automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`). Do not paste a role body into a generic worker, explorer, or other host agent at runtime.
- If the host cannot expose one of the named agents (configuration disabled, permission denied, capability missing), stop under SUBAGENT-PROTOCOL.md's "Host does not expose subagent support" condition. Do not fall back to runtime-curated prompt injection.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
