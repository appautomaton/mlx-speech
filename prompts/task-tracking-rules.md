# Task Tracking Rules

Use the todo/task-list tool as your working memory throughout the session.

## Creating the list

Read the relevant source files, plan, and references before writing any
items. Understand the real scope first. Each item must be one verifiable
action — a file created, a function implemented, a test passing. Never
write phase-level items like "implement the tokenizer." Break those into
the actual steps.

## During execution

Each tool call replaces the full list. Use that: reshape the list every
time you update it. The initial list will be incomplete — that is
expected.

You MUST:
- Add items the moment you discover new work a step requires.
- Break down items that turn out to be larger than one action.
- Remove items that are no longer needed.
- Reorder by actual dependency, not original position.
- Keep exactly one item in_progress at a time.
- Mark items complete only when acceptance criteria are met.

Do not wait until a step is finished to update the list. Update it as
your understanding changes.

## Working through the list

Own the work. When you finish an item, verify it, then move to the next
immediately. Read the code, check the implementation, and keep going —
do not pause to report back after each step.

Use your best judgment to make decisions as you go. When you encounter
ambiguity or conflicting information, investigate: read the source
truth, check upstream references, and search online for authoritative
and up-to-date sources when local context is insufficient. Make the
call, document your reasoning briefly, and continue.

If a step is blocked, skip it, note the blocker, and continue with the
next unblocked item. Only stop when: every item is done, a red flag
requires user input, or all remaining items are blocked.
