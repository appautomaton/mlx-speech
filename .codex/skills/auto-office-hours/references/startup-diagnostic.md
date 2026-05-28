# Startup Mode: Six Forcing Questions

Ask one at a time. Push until the answer names concrete evidence, a specific stakeholder, or an observable workaround. If the answer remains category-level after the allowed pushes, use the STOP conditions in SKILL.md instead of continuing.

## Questions

**Q1: Demand Reality.** "What's the strongest evidence that someone actually wants this, not 'is interested,' but would be genuinely upset if it disappeared tomorrow?"

Push target: specific behavior, someone paying, someone building their workflow around it.

**Q2: Status Quo.** "What are your users doing right now to solve this problem, even badly? What does that workaround cost them?"

Push target: a specific workflow, hours spent, tools duct-taped together.

**Q3: Desperate Specificity.** "Name the actual human who needs this most. What's their title? What gets them promoted? What gets them fired?"

Push target: a name, a role, a specific consequence they face if the problem isn't solved.

**Q4: Narrowest Wedge.** "What's the smallest possible version of this that someone would pay real money for, this week, not after you build the platform?"

Push target: one feature, one workflow, something shippable in days.

Scope note: This question tests shippability instinct, not scope. Use the answer to understand what the user considers the core value, then return to their stated goal. Do not replace a capability-sized goal with the narrowest wedge answer.

**Q5: Observation & Surprise.** "Have you watched someone use this without helping them? What did they do that surprised you?"

Push target: a specific surprise that contradicted the founder's assumptions.

**Q6: Future-Fit.** "If the world looks meaningfully different in 3 years, does your product become more essential or less?"

Push target: a specific claim about why the product becomes more valuable as the world changes.

## Smart Routing by Product Stage

- Pre-product → Q1, Q2, Q3
- Has users → Q2, Q4, Q5
- Has paying customers → Q4, Q5, Q6
- Pure engineering/infra → Q2, Q4 only

## Smart Routing by Scope Classification

- Bug-sized → Q2 only (status quo / workaround cost), then move to alternatives
- Feature-sized → standard routing by product stage
- Capability-sized → Q1, Q2, Q5 (demand, status quo, observation). Use Q4 as a calibration probe to understand the core value, not to set scope.
- Roadmap-sized → Q1, Q2, Q3, then help decompose into the first spec candidate

## Shape-Aware Routing

When the shape is not feature, supplement or replace the scope-routed questions above. Shape questions take priority; mode questions fill remaining gaps.

- **Parity** → Q2 (status quo) stays relevant. Add: What is the reference system? Which gaps affect paying customers most? What does parity look like — matching output, passing compliance, meeting a benchmark?
- **Audit** → Q1 (demand reality) stays relevant. Add: What questions must the audit answer? What decision depends on the findings? What evidence would change your next move?
- **Refactor** → Q2 (status quo) stays relevant. Add: What structural problem are you solving? What behavior must remain invariant? What's the blast radius?
- **Migration** → Q2 (status quo) + Q4 (narrowest wedge) stay relevant. Add: What's the target state? What compatibility constraints exist? What's the rollback plan?
- **Coverage** → Q5 (observation / surprise) stays relevant. Add: What risk areas are undertested? Where have bugs escaped? What's the target improvement?
- **Mixed** → Combine the questions from each constituent shape. Ask the highest-priority question from each.
