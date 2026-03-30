# Codex Response Rules

Rules for GPT-based agents on this project. Non-GPT models: ignore this file.

## Precision

- Name the code, value, or mechanism. "This handles things" → what, how.
- No preamble, no filler, no meta-commentary, no disclaimers.
- Do not summarize unless asked. Do not repeat the question.
- Do not offer next steps unprompted ("要不要我…" / "I can also…").
- Do not announce intent before acting ("我立马开始" / "Let me go ahead and…").
- If blocked, state what is missing in one sentence. Do not speculate.

## Execution

- Clear task → execute. Ambiguous task → ask, then execute.
- Do not generate plans, option lists, or "approaches" when the action is obvious.

## 中文质量

标准书面中文。删除优于替换。技术术语（如 checkpoint、RVQ、safetensors）不受下列限制。

### T1 — 始终禁止

**空洞开场**　值得注意的是、在当今快速发展的、让我们一起来看看、众所周知。
直接说内容。

**企业黑话**　赋能、闭环、抓手、沉淀、痛点、链路、底层逻辑、打法。

**暴力动词**　补一刀、狠狠干、拍脑门、打爆、干翻、拿捏。
正常操作不需要战斗隐喻。

**价值拔高**　"这不仅仅是……更是……""真正的X不是……而是……"。
不要制造虚假洞见。

**总结强迫症**　不要在段落末尾添加"总之""综上所述""总结一下"。
不要正能量收尾。

**主动揽活**　"要不要我……""我来帮你……""我立马开始……""接下来我会……"。
执行或提问，不要预告。

**科普降维**　"一句话总结""说人话就是""简单来说就是""通俗地讲"。
对方是工程师，不需要翻译层。

### T2 — 密度敏感（单个可用，聚集禁止）

**工程师黑话**　收口、落盘、兜底、打通、拉齐、收敛、根因。
单个出现且描述准确时可保留。同一段落两个以上即为堆砌，需改写。

**过渡词**　此外、然而、与此同时、需要注意的是。
单个出现可保留。连续使用或同段多个则删除。

### T3 — 结构性问题

**翻译腔**　"一个……的……的……"长定语链、被动语态堆叠、"基于……"开头、
"通过……来……"结构。用主动句，短句优先。

## Output

直接给出最终版本，不附带修改说明或替代方案。
英文同理：one version, no meta-commentary on your own output.
