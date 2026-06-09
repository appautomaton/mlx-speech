# Slice Design Examples

## Well-Designed Slice

```markdown
### Slice 3: Implement user authentication middleware

**Objective:** Add JWT validation to all protected API routes.
**Acceptance criteria:**
- Protected API routes reject missing or invalid JWT Bearer tokens.
- Existing valid-token requests still succeed.
**Touches:** `src/middleware/auth.js`, protected route handlers, auth middleware tests.
**Produces:** `src/middleware/auth.js` with `verifyToken` function, plus updated route handlers.
**Verification:** `npm test -- auth.middleware.test.js` passes; `curl -H "Authorization: Bearer invalid" /api/protected` returns 401.
```

## Poorly-Designed Slice

```markdown
### Slice 3: Add auth

**Objective:** Make the API secure.
**Produces:** Working authentication.
**Verification:** Tests pass.
```

## Subagent-Routed Slice

```markdown
### Slice 4: Migrate session auth to JWT across API routes

**Objective:** Replace session-based auth with JWT validation on all protected endpoints.
**Acceptance criteria:**
- All protected routes validate JWT Bearer tokens
- Session cookie auth is removed, not left as fallback
- Existing auth tests pass with JWT tokens
**Verification:** `npm test -- auth` passes; `curl -H "Authorization: Bearer <valid>" /api/protected` returns 200; `curl --cookie "session=old" /api/protected` returns 401.
**Execution:** subagent recommended
**Touches:** `src/middleware/auth.js`, `src/routes/api/users.js`, `src/routes/api/settings.js`, `src/utils/jwt.js`, `tests/auth.test.js`
```

Why subagent recommended: touches 5 files across middleware, routes, and utilities; crosses subsystem boundaries with shared interface changes.

## Topology Section

A PLAN.md topology section names the default route, then only the overrides:

```markdown
## Execution Routing and Topology

Default: direct, serial, continuation after verification.

Overrides:
- Slice 4: subagent recommended (5 files across auth and routing subsystems)

Parallel-safe groups:
- Slices 2 and 3 (disjoint write sets: Slice 2 touches `src/db/migrations/`, Slice 3 touches `src/ui/components/`; no shared state)

Checkpoints:
- Slice 6: human-verify (visual layout review, not automatable)
```

## Rule of Thumb

If you cannot write the verification command before starting the slice, the slice is not well-defined.
