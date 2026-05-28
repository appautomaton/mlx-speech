# Builder Mode: Design Partner

Ask one at a time. The goal is to brainstorm and sharpen, not interrogate.

## Questions

- **Q1: What's the coolest version of this?** What would make it genuinely delightful?
- **Q2: Who would you show this to?** What would make them say "whoa"?
- **Q3: What's the fastest path to something you can actually use or share?** Use the answer to understand what the user considers demonstrable progress, then return to their full goal. Do not redirect the conversation to the fast path if the user brought a larger vision.
- **Q4: What existing thing is closest to this, and how is yours different?**
- **Q5: What would you add if you had unlimited time?** What's the 10x version?

## Smart Routing by Scope Classification

- Bug-sized → Q3 (fastest path) + Q4 (how is yours different?), then move to alternatives
- Feature-sized → standard (all five questions)
- Capability-sized → Q1 (coolest version), Q4 (how is yours different?), Q5 (10x version)
- Roadmap-sized → Q1 (coolest version), Q2 (who would you show this to?), then decompose into the first spec candidate

## Shape-Aware Routing

When the shape is not feature, supplement or replace the scope-routed questions above. Shape questions take priority; mode questions fill remaining gaps.

- **Parity** → What is the reference system or benchmark? Which areas have the most gaps? What does "closed" look like — matching output, passing tests, meeting a performance bar? Which gaps are safe to close now vs. blocked?
- **Audit** → What questions must the audit answer? What evidence would change the next decision? What's the decision gate — what happens with the findings?
- **Refactor** → What structural problem are you solving? What behavior must remain invariant? What's the blast radius? How do you verify nothing broke?
- **Migration** → What's the source state and target state? What compatibility constraints exist? What's the rollback plan?
- **Coverage** → What risk areas are undertested? What's the target improvement? What types of tests are needed?
- **Mixed** → Combine the questions from each constituent shape. Ask the highest-priority question from each.
