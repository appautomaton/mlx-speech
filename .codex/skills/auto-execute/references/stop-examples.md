# Stop Condition Examples

## When to Halt

**Missing dependency:** The plan requires `libpq` but the system has no PostgreSQL client libraries and the user has no admin rights to install them. Halt. Do not attempt a workaround that compromises the environment.

**Repeated test failure:** A test fails 3 times with the same error. The error is not a typo; it indicates a structural mismatch between the test expectation and the implementation. Halt. Investigate before attempting a fourth fix.

**Ambiguous plan instruction:** The plan says "refactor the auth module" but the auth module is 5,000 lines across 12 files with no specified target state. Halt. Ask for a more specific slice.

**Stale plan:** The plan references `src/handlers/user.js` but that file was renamed to `src/handlers/account.js` in a commit since the plan was written. Halt. The plan needs refresh.

**Scope creep:** The user says "while you're in there, can you also add OAuth?" mid-slice. Halt. Reframe: record as follow-up or revisit the plan.

## When to Push Through

**Typo in test:** The test fails because of a clear typo in the assertion (`expect(foo).toBe(bar)` where `bar` is obviously wrong). Fix the typo and continue.

**Lint failure:** The build fails because of a missing semicolon or import order. Fix and continue.

**Expected dependency conflict:** The plan warned that two packages have conflicting peer dependencies and specified the resolution strategy. Follow the strategy and continue.

**Flaky test:** A test fails intermittently and the plan explicitly notes it as flaky with a workaround. Apply the workaround and continue.

## Decision Rule

If the obstacle is **trivial** (typo, lint, known flaky test) → fix and continue.
If the obstacle is **structural** (missing dependency, ambiguous instruction, stale plan) → halt and report.
If you are unsure, run one bounded diagnostic. If the obstacle is still structural or ambiguous, halt and report.
