# Risk Matrix Examples

## Example 1: API Migration

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Architecture fit | 8 | Uses existing service layer; no new abstractions |
| Data flow clarity | 6 | Most flows traced; one batch job path is unclear |
| Edge case coverage | 4 | Timeout and retry handled; partial failure not modeled |
| Test strategy | 5 | Unit tests specified; integration tests missing |
| Rollback safety | 7 | Feature flag wrapped; database migration is backward-compatible |
| Dependency risk | 9 | No new dependencies; uses existing HTTP client |

**Verdict:** `approved_with_risks`. Edge case coverage and test strategy need attention before merge.

## Example 2: New Feature with External Service

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Architecture fit | 5 | New service boundary; overlaps with existing notification system |
| Data flow clarity | 7 | Webhook flow documented; retry logic specified |
| Edge case coverage | 3 | External service downtime not handled; rate limiting unknown |
| Test strategy | 4 | Mock-based unit tests; no contract tests with external service |
| Rollback safety | 6 | Feature flag present; but data written to external service cannot be reverted |
| Dependency risk | 2 | New third-party service with no SLA; no fallback defined |

**Verdict:** `needs_correction`. Dependency risk and edge case coverage are blocking.
