# Engineering Review Sections

Use this as a trigger-based risk checklist. Do not run every section by default, and do not write "No issues found" filler. Evaluate only sections whose trigger appears in PLAN.md, DESIGN.md, or the changed surface. Summarize clean sections in one line at most; expand only verdict-driving risks.

## Section 1: Architecture Review

Trigger: new pattern, cross-module integration, state machine, pipeline, external service, or unclear component boundary.

Evaluate when triggered:
- Overall system design and component boundaries. Dependency graph.
- Data flow: happy path and material shadow paths. Add one diagram only when prose would be ambiguous.
- State machines. Diagram only non-trivial stateful objects. Include impossible/invalid transitions when they affect risk.
- Coupling concerns. Before/after dependency graph.
- Scaling characteristics. What breaks first under 10x load? Under 100x?
- Single points of failure. Map them.
- Security architecture. Auth boundaries, data access patterns, API surfaces.
- Production failure scenarios. One realistic failure per new integration point.
- Rollback posture. Git revert? Feature flag? DB migration rollback? How long?

## Section 2: Error & Rescue Map

Trigger: new error handling, external calls, persistence, retries, parsing, async jobs, or user-visible failure states.

For risk-bearing methods/services, fill in only rows that can change the verdict:

```
  METHOD/CODEPATH          | WHAT CAN GO WRONG           | EXCEPTION CLASS
  -------------------------|-----------------------------|-----------------
  ExampleService#call      | API timeout                 | TimeoutError
                           | API returns 429             | RateLimitError

  EXCEPTION CLASS              | RESCUED?  | RESCUE ACTION          | USER SEES
  -----------------------------|-----------|------------------------|------------------
  TimeoutError                 | Y         | Retry 2x, then raise   | "Service temporarily unavailable"
  JSONParseError               | N ← GAP   | —                      | 500 error ← BAD
```

Rules:
- Catch-all error handling is ALWAYS a smell. Name specific exceptions.
- Every rescued error must: retry with backoff, degrade gracefully, or re-raise with context.
- "Swallow and continue" is almost never acceptable.
- For each GAP: specify rescue action and what the user should see.

## Section 3: Security & Threat Model

Trigger: new user input, auth/permission boundary, secrets, file paths, network calls, dependencies, sensitive data, or injection surface.

Evaluate when triggered:
- Attack surface expansion. New endpoints, params, file paths, background jobs?
- Input validation. For every new user input: nil, empty, wrong type, max length, injection attempts.
- Authorization. Scoped to right user/role? Direct object reference vulnerabilities?
- Secrets and credentials. In env vars, not hardcoded? Rotatable?
- Dependency risk. New packages? Security track record?
- Data classification. PII, payment data? Handling consistent with existing patterns?
- Injection vectors. SQL, command, template, LLM prompt injection.
- Audit logging. For sensitive operations: is there an audit trail?

For each finding: threat, likelihood (High/Med/Low), impact (High/Med/Low), mitigation status.

## Section 4: Data Flow & Interaction Edge Cases

Trigger: new data transform, persistence, UI interaction, workflow state, or user-visible async behavior.

**Data Flow Tracing:** For complex new data flow, one diagram may help:
```
  INPUT ──▶ VALIDATION ──▶ TRANSFORM ──▶ PERSIST ──▶ OUTPUT
    │            │              │            │           │
    ▼            ▼              ▼            ▼           ▼
  [nil?]    [invalid?]    [exception?]  [conflict?]  [stale?]
  [empty?]  [too long?]   [timeout?]    [dup key?]   [partial?]
  [wrong    [wrong type?] [OOM?]        [locked?]    [encoding?
   type?]                                                     ]
```

**Interaction Edge Cases:** For new user-visible interactions, evaluate only applicable edge cases: double-click, stale state, navigate-away, slow connection, zero results, 10,000 results, retry-while-in-flight.

Flag any unhandled edge case as a gap. For each gap, specify the fix.

## Section 5: Code Quality Review

Trigger: always, but keep it bounded to the files and patterns touched by the plan.

- Code organization and module structure. Fits existing patterns?
- DRY violations. Be aggressive. Reference file and line.
- Naming quality. Named for what they do, not how.
- Error handling patterns. (Cross-reference Section 2.)
- Missing edge cases. List explicitly.
- Over-engineering check. Any abstraction solving a problem that does not exist yet?
- Under-engineering check. Fragile? Happy-path only?
- Cyclomatic complexity. Flag any method with >5 branches.

## Section 6: Test Review

Trigger: always. Use a compact coverage map, not a diagram dump.

Map new things when present:
- NEW UX FLOWS
- NEW DATA FLOWS
- NEW CODEPATHS
- NEW BACKGROUND JOBS / ASYNC WORK
- NEW INTEGRATIONS / EXTERNAL CALLS
- NEW ERROR/RESCUE PATHS

For each item:
- What type of test covers it? (Unit / Integration / System / E2E)
- Does a test exist in the plan?
- Happy path test?
- Failure path test? (Be specific.)
- Edge case test? (nil, empty, boundary, concurrent access)

Test ambition check:
- What test would make you confident shipping at 2am on a Friday?
- What test would a hostile QA engineer write to break this?
- What is the chaos test?

Test pyramid check: Many unit, fewer integration, few E2E? Or inverted?
Flakiness risk: Flag tests depending on time, randomness, external services, or ordering.

## Section 7: Performance Review

Trigger: new queries, loops over unbounded data, background jobs, large files, caching, concurrency, or external calls.

- N+1 queries. Every new association traversal: includes/preload?
- Memory usage. Maximum size in production for every new data structure.
- Database indexes. Every new query: is there an index?
- Caching opportunities. Every expensive computation or external call.
- Background job sizing. Worst-case payload, runtime, retry behavior.
- Slow paths. Top 3 slowest new codepaths and estimated p99 latency.
- Connection pool pressure. New DB, Redis, HTTP connections?

## Section 8: Observability & Debuggability

Trigger: operated service behavior, production workflows, async jobs, external dependencies, sensitive flows, or hard-to-debug state.

- Logging. Structured log lines at entry, exit, and each significant branch?
- Metrics. What metric tells you it is working? What tells you it is broken?
- Tracing. Trace IDs propagated for cross-service flows?
- Alerting. What new alerts should exist?
- Dashboards. What new panels on day 1?
- Debuggability. Can you reconstruct what happened from logs alone?
- Admin tooling. New operational tasks that need admin UI or scripts?
- Runbooks. For each new failure mode: what is the operational response?

## Section 9: Deployment & Rollout

Trigger: deployment risk, migrations, feature flags, config/env changes, data compatibility, or irreversible state.

- Migration safety. Backward-compatible? Zero-downtime? Table locks?
- Feature flags. Should any part be behind a flag?
- Rollout order. Migrate first, deploy second?
- Rollback plan. Explicit step-by-step.
- Deploy-time risk window. Old code + new code simultaneously: what breaks?
- Environment parity. Tested in staging?
- Post-deploy verification. First 5 minutes? First hour?
- Smoke tests. What automated checks immediately post-deploy?

## Section 10: Long-Term Trajectory

Trigger: architectural commitment, new dependency, durable schema/API, framework choice, or path-dependent abstraction.

- Technical debt introduced. Code, operational, testing, documentation debt.
- Path dependency. Does this make future changes harder?
- Knowledge concentration. Documentation sufficient for a new engineer?
- Reversibility. Rate 1-5: 1 = one-way door, 5 = easily reversible.
- Ecosystem fit. Aligns with ecosystem direction?
- The 1-year question. Read this plan as a new engineer in 12 months: obvious?

## Section 11: Design & UX Review

Skip only if no UI scope detected.

- Information architecture: what does the user see first, second, third?
- Interaction state coverage map: LOADING | EMPTY | ERROR | SUCCESS | PARTIAL
- User journey coherence: storyboard the emotional arc
- AI slop risk: does the plan describe generic UI patterns?
- DESIGN.md alignment: does the plan match the stated design system?
- Responsive intention: mobile mentioned or afterthought?
- Accessibility basics: keyboard nav, screen readers, contrast, touch targets

Add a user-flow diagram only when screens or states are complex enough that prose would be ambiguous.
