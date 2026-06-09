# Implementer Dispatch Packet

Per-call dispatch slots for the `automaton-implementer` subagent. The static role body (identity, boundaries, self-review, return envelope) is installed as the host-native agent definition. The coordinator fills these slots from the active slice and hands them to the named agent. On a re-dispatch after a reviewer returns `CHANGES_REQUESTED`, the coordinator also fills `<requested-changes>` with the reviewer's concrete issues. On a first dispatch that slot stays empty and can be left out.

```text
<slice>
{SLICE_TEXT}
</slice>

<constraints>
{CONSTRAINTS}
</constraints>

<acceptance-criteria>
{ACCEPTANCE_CRITERIA}
</acceptance-criteria>

<requested-changes>
{REQUESTED_CHANGES}
</requested-changes>
```
