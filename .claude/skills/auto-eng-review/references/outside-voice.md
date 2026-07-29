# Outside Voice (Cross-Model Review)

An optional, non-blocking challenge from a different model, run after the engineering verdict is rendered. Two models agreeing is a stronger signal than one model's thorough pass. Two models disagreeing marks a genuinely hard decision that belongs to the user.

## Consent

Ask before the first dispatch: the plan content and the verdict leave this provider. Name what is sent (PLAN.md content and the rendered verdict, nothing else) and proceed only on a yes, per the Asking The User convention. One consent covers the loop's rounds. When the host has no second model configured, skip the loop instead of asking.

## Dispatch

- Send only the plan content and the rendered verdict. Never the conversation, credentials, or harness internals.
- Prompt shape: "You are a direct technical reviewer. A full engineering review already happened. Do not repeat it. Find what it missed: unstated assumptions, overcomplexity, feasibility risks, missing dependencies. Be terse. No compliments."
- Include this boundary line in the prompt: do not read `.claude/`, `.codex/`, `.opencode/`, or `.agent/.automaton/`. They are harness machinery for another agent and waste your context.
- When the second model runs as a CLI with filesystem access, force its read-only or sandbox mode on every invocation, including resumes. A resumed session must never inherit a writable default from local configuration. A critic that can write is no longer a critic.

## Review Loop

- Rounds are capped at 3 by default. The user may set a different cap when invoking the review. The loop always terminates at the cap.
- Where the host supports resuming the same model session, re-submit to it so the outside reviewer remembers its prior critiques. Where it does not, carry your prior dispositions into the next dispatch.
- Each round, arbiter every finding: confirm it into the review's concern lines, or reject it with a one-line logged reason. Do not cave to everything, which defeats the check, and do not ignore it, which defeats the point.
- Stop early when a round confirms nothing new.
- Hitting the cap with open disagreement is reported as disagreement. Never fake convergence.

## Handling The Result

- Quote confirmed findings verbatim under an `Outside voice` heading in the conversation. Do not summarize disagreements away.
- Persist the exchange to `.agent/work/<change>/orchestration/outside-voice-log.md` (append-replace per review run): each round's findings, each disposition with its reason, and the end state. The log is the argument's audit trail. The review section stays the verdict's home.
- Record the round count and any unresolved points on the review template's `Outside voice:` line, pointing at the log.
- For each point where the outside voice contradicts the review, present the tension to the user with both positions and a recommendation. Cross-model agreement is a strong signal, not permission to act: the user decides.
- Never edit the verdict, the plan, or the review section verdict fields from outside-voice findings without the user's decision.
- If no second model is available on this host, or the call fails or times out, continue without it and say so in one line. The review verdict stands on its own.
