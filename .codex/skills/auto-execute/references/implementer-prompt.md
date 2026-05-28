# Implementer Dispatch Packet

Per-call dispatch slots for the `automaton-implementer` subagent. The static role body (identity, boundaries, self-review, return envelope) lives in `implementer-role.md` and is installed as the host-native agent definition; the coordinator fills these slots from the active slice and hands them to the named agent.

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
```
