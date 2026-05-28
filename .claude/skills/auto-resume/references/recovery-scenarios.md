# Recovery Scenarios

## Scenario 1: Fresh Session, Active Change Exists

**State:** `current.json` has `active_change: "feature-x"`, `stage: "execute"`.
**Action:** Load SPEC.md, DESIGN.md, PLAN.md. Identify current slice. Summarize and recommend `auto-execute`.

## Scenario 2: Fresh Session, No Active Change

**State:** `current.json` has `active_change: "none"` or file is missing.
**Action:** Check if `.agent/` exists. If yes, read steering artifacts that exist and ask user what to work on. If no, recommend `auto-onboard`.

## Scenario 3: Stale Canonical Pointer

**State:** `current.json` points to `.agent/work/feature-x/SPEC.md` but file does not exist.
**Action:** Report stale pointer. Search `.agent/work/` for existing artifacts. If found, ask user to confirm. If not found, recommend `auto-frame`.

## Scenario 4: Review Verdict Blocks Progress

**State:** `current.json` has `product_review: "needs_clarification"` but stage is `plan`.
**Action:** Surface the review verdict. Recommend `auto-frame` to address the clarification before planning.

## Scenario 5: Scaffold-Level Steering

**State:** Steering artifacts exist but contain only scaffold placeholders. No real project truth.
**Action:** Recommend `auto-onboard`. Do not proceed with execution on scaffold-level steering.

## Scenario 6: Multiple Changes in Progress

**State:** `.agent/work/` contains multiple change directories.
**Action:** List them. Ask user which to resume. Do not guess.

## Stage Routing

- Stage `frame` with no SPEC.md: recommend `auto-frame`.
- Stage `frame` with SPEC.md: recommend `auto-plan`; mention `auto-ceo-review` only when product direction needs review.
- Stage `plan` with no PLAN.md: recommend `auto-plan`.
- Stage `plan` with PLAN.md: recommend `auto-execute`; mention `auto-eng-review` only when execution safety needs review.
- Stage `execute`: recommend `auto-execute`.
- Stage `verify` → `auto-verify`.
- Stage `verified` → change complete; report completion.
- Stage `resume` with missing steering: recommend `auto-onboard`.
- Change complete and ROADMAP.md has pending items: surface them as optional future work; no next lifecycle skill by default.
- Change complete and no pending roadmap items: `none - change complete`.
