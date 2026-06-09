# Verify Quality

Load this reference only before writing the final verification report.

## VERIFY Anti-Patterns

- Completion theater: "all good" without commands, outputs, and acceptance criteria.
- Stale proof: relying on prior execution evidence instead of fresh verification.
- Hidden skipped checks: omitted commands not called out as gaps.
- Partial-pass language: softening FAIL or PARTIAL into "mostly working".
- Generic confidence: claims that do not map to a test, file, or observed behavior.

## Better Shape

- Tie every PASS, FAIL, or PARTIAL result to a command or direct observation.
- Name skipped checks and why they were skipped.
- Compare actual evidence to the plan's acceptance criteria.
- Recommend the next skill based on evidence, not optimism.

## Prose Hygiene

Verification reports attract hedging and confidence inflation. Results should read like lab notes, not press releases.

Scan for:
- "mostly working", "largely complete": use PASS, FAIL, or PARTIAL
- "appears to", "seems to": either it does or you need another test
- "successfully verified" without naming the command and output
- Generic positive conclusions ("overall the implementation is solid")
- Promotional language about test coverage

Before: "The authentication system appears to be working well. Tests largely pass and the overall implementation seems solid."
After: "PASS: `npm test -- auth.test.js`, 12/12 assertions. FAIL: `curl -H 'Authorization: Bearer expired' /api/me`, returns 200, expected 401."

## Final Check

If a reader cannot reproduce the verification from the artifact, revise it.
