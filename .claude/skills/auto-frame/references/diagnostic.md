# Framing Diagnostic

Load this only when Choose Depth says the request needs conversation before a spec.

Ask only questions that make the objective frameable. Never ask what the repo can answer: explore or dispatch the librarian first. Follow the Asking The User convention in `.agent/.automaton/references/FRAMEWORK.md`.

The routing below is the contract. The wording is yours: ask each topic as your own question, in the user's language.

## Mode Routing

**Startup mode** when demand, user, market, or customer evidence matters. Topics: demand evidence, the status quo workaround and its cost, the specific person who needs this, the narrowest wedge, unassisted observation, future fit.

| Product stage | Topics |
| --- | --- |
| Pre-product | demand, status quo, specific person |
| Has users | status quo, wedge, observation |
| Has paying customers | wedge, observation, future fit |
| Pure engineering or infrastructure | status quo, wedge |

**Builder mode** when the work is personal, exploratory, open-source, or design-partner shaped. Topics: the coolest version, who they would show it to, the fastest usable path, the closest existing thing and how theirs differs, the 10x version.

**Content mode** when the deliverable is prose: read `content-framing.md`. Content is a peer mode alongside Startup and Builder, not an overlay on them.

**Any mode:** read `landscape-awareness.md` when market, ecosystem, competitor, or current-state evidence would change the frame. Its consent gate governs every outbound search.

## Scale Routing

| Work scale | Startup topics | Builder topics |
| --- | --- | --- |
| bug | status quo, then alternatives | fastest path, difference |
| feature | route by product stage | all five |
| capability | demand, status quo, observation | coolest version, difference, 10x |
| roadmap | demand, status quo, specific person, then decompose | coolest version, audience, then decompose |

The wedge and fastest-path topics probe what the user treats as core value. They do not set scope. Do not replace a capability-sized goal with the answer to a smallest-version question, and do not redirect a user who brought a larger vision onto the fast path.

When the shape is not feature, shape questions take priority and the routing above fills the remaining gaps: read `shape-questions.md`.

## Follow-Up Discipline

Follow up when an answer changes scope, reveals a constraint, contradicts earlier context, or stays abstract. Ask for a concrete correction or choice, not a generic reaction. If the answer is polished but vague, push until it names concrete evidence, a specific stakeholder, or an observable workaround.

Read `diagnostic-calibration.md` when the diagnostic feels soft or agreeable rather than evidence-backed.

## Grill Mode

Grill mode deepens the diagnostic from minimum to exhaustive: walk each branch of the decision tree, resolving dependent decisions one at a time, until shared understanding is reached or the user calls it done.

It starts two ways only: the user asks, or the SKILL's depth offer is accepted. That offer fires when the open questions reach three, when the work is high-stakes (auth, schema, concurrency, migration, payments), or when the work is roadmap-sized. The user accepts or declines. Never self-escalate into a grill.

Read `diagnostic-calibration.md` on entry: it carries the depth mechanics this mode runs on.

Ask exactly one question per call here, whatever the host tool permits. Each answer reshapes which branch comes next, so a batch spends questions the previous answer would have rewritten.

## Alternatives

Present 2-3 distinct approaches matching the work scale and shape, using `alternatives-format.md`. Make them differ by scope, risk, learning value, traceability, or verification strength, never by tone.

Recommend one and evaluate the evidence directly: name what supports it, what it does not prove, and what evidence would change the recommendation.

Return to Cover The Request once the user picks an approach. The spec is written there, from the approved wording.
