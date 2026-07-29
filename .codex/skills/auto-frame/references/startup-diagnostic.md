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

## Smart Routing by Work Scale

- Bug-sized → Q2 only (status quo / workaround cost), then move to alternatives
- Feature-sized → standard routing by product stage
- Capability-sized → Q1, Q2, Q5 (demand, status quo, observation). Use Q4 as a calibration probe to understand the core value, not to set scope.
- Roadmap-sized → Q1, Q2, Q3, then help decompose into the first spec candidate

## Shape-Aware Routing

When the shape is not feature, read `references/shape-questions.md`: shape questions take priority, and the scope-routed questions above fill the remaining gaps.
