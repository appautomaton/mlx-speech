# Debug Protocol

Bounds a diagnosis when the fix is not obvious, so investigation ends in a report rather than an open-ended search.

## Escalation

Escalate if you cannot isolate the root cause within 3 attempts: three failed hypotheses mean the mental model is wrong, and further attempts only spend budget confirming that. Report:

```
**Observed:** [what the system does]
**Expected:** [what the system should do]
**Tried:** [what you investigated]
**Need:** [what you need from the user to proceed]
```
