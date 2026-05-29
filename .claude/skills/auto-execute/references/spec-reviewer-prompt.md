# Spec Reviewer Dispatch Packet

Per-call dispatch slots for the `automaton-spec-reviewer` subagent. The static role body (identity, boundaries, check list, return envelope) is installed as the host-native agent definition; the coordinator fills these slots after an implementer reports `DONE` or acceptable `DONE_WITH_CONCERNS`.

```text
<slice>
{SLICE_TEXT}
</slice>

<acceptance-criteria>
{ACCEPTANCE_CRITERIA}
</acceptance-criteria>

<implementation-summary>
{IMPLEMENTATION_SUMMARY}
</implementation-summary>
```
