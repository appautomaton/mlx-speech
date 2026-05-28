# Quality Reviewer Dispatch Packet

Per-call dispatch slots for the `automaton-quality-reviewer` subagent. The static role body (identity, boundaries, severity labels, check list, return envelope) lives in `quality-reviewer-role.md` and is installed as the host-native agent definition; the coordinator fills these slots only after spec compliance is `APPROVED`.

```text
<slice>
{SLICE_TEXT}
</slice>

<implementation-summary>
{IMPLEMENTATION_SUMMARY}
</implementation-summary>
```
