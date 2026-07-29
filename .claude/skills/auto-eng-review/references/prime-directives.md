# Engineering Prime Directives

The standing creed for every engineering review. The trigger-based checklist in `engineering-sections.md` owns the per-risk questions. These directives own the posture.

1. **Zero silent failures.** Every failure mode must be visible to the system, the team, or the user. A silent failure is a critical defect wherever it appears.

2. **Specificity is the review.** Name the exception, the file, the test, the metric. "Handle errors" and "add tests" are not findings. A finding the implementer cannot act on is not a finding.

3. **Observability is proportional to scope.** New codepaths need a way to diagnose failure. Dashboards, alerts, and runbooks become first-class only when the plan changes operated production behavior.

4. **Deferred work needs an owner surface.** Record deferred work only in the approved plan or a review action someone downstream will act on. Roadmap phases are not a review-time surface, and TODO files are not a default.

5. **You have permission to say "scrap it and do this instead."** A fundamentally better approach tabled late costs more than a hard conversation now.

6. **Diagrams earn their space.** A plan that adds a diagram must need it (prose would be ambiguous) and must maintain it: a stale diagram is worse than none.
