# TNT live consumer for R2T2 streaming

**Status: PARKED — not started and not assigned.**

Begin only when the user schedules it, after the R2T2 streaming performance
work in `confucius4-r2t2-live-cost.md`. The agent executing that plan does
not execute this one. TNT code lives in `/Users/ac/dev/ai/tnt-asr` and
follows that repository's `AGENTS.md`. Library prerequisites below are
mlx-speech changes scheduled together with the TNT work.

## Current state (verified 2026-09-26)

- TNT branch `r2t2-streaming` has uncommitted changes in `app.py`,
  `audio.py`, `avf_audio.py`, `transcriber.py`, widgets, tests,
  `pyproject.toml`, and `CLAUDE.md`. Preserve them.
- The live path runs only when `TNT_MLX_MODEL` points to the R2T2 package.
  The default `bin/qwen3-asr-mlx` link targets a Qwen3-ASR build with no
  `stream_session`. The local mlx-speech path override must not ship on
  `master` (TNT `AGENTS.md`).
- `LiveUtterance.push_int16` (`src/tnt/transcriber.py:213`) calls `feed`,
  which processes every ready window before returning. When steps run
  slower than audio, the UI gets no update until the backlog clears, and
  the next call inherits a larger backlog.
- `pump_live` (`transcriber.py:231`) sleeps on empty pulls (`:244`). This
  is harmless while `feed` drains everything; it becomes a defect once
  `feed` is bounded.
- `_finish_live` (`src/tnt/app.py:467`): a `begin_stop` timeout or
  exception is swallowed (`:478`), so the final capture boundary is
  undefined. The result of `thread.join(60.0)` (`:488`) is not checked; the
  recorder is then cleared or recreated, and `_live_final or _live_text` is
  shown as the final transcript even if the worker is still running.

## Library prerequisites (mlx-speech)

- `feed(pcm, *, max_steps: int | None = None)`. The default keeps current
  behavior; a positive bound limits processed windows; `feed(empty,
  max_steps=1)` advances queued windows. Expose `pending_samples` and
  `has_pending_step` (a complete current window is ready, including the
  larger first window) on the adapter session. Aggregate `delta` across
  processed windows. Empty input with no ready window does no inference.
- `finalize()` drains all complete windows with normal step semantics, then
  processes the short tail once with finish semantics. No tail releases the
  held-back text without a model call. It must work after a single
  whole-file `feed` and after a bounded feed with pending windows.
- An utterance shorter than one hop (160 samples) yields zero mel frames
  today. Define a minimum one-hop feature pad for that final case, keeping
  the real sample count. Cover 0, 1, 159, and 160 samples; empty sessions
  finish without inference.
- Successful finalization is idempotent and releases session state; later
  `feed` calls raise a closed-session error. An inference failure marks the
  session failed, keeps committed text as partial output, and never retries
  silently with partially advanced state.
- Sessions sharing one model keep isolated state.
- Tests: one-shot, fragmented, and bounded-plus-drain feeding produce the
  same decoder calls, budgets, commits, and final result.

## TNT changes

Respect TNT's non-negotiables: no recorder calls on the UI thread; 1 s stop
and 3 s start timeouts, with timed-out recorders abandoned and rebuilt;
in-process MLX work is abandoned, never killed; `uv` only; the `os._exit`
path stays as is.

- Pump one window per iteration, publish each committed update, and drain
  ready windows when the recorder returns no new PCM.
- On stop, establish the final captured-sample boundary and transfer every
  sample through it exactly once before the recorder is cleared or reused.
  Drain complete windows with progress shown, then finalize the tail.
- No silent success. If the drain is unfinished at a timeout, show
  processing or failure explicitly and keep ownership until completion or
  user cancellation. With bounded steps, `Space` cancellation takes effect
  between windows.
- If `begin_stop` times out, report the capture boundary as uncertain
  instead of presenting the transcript as complete.
- No duplicate full-take `generate` and no timeout that drops captured
  audio.
- Tests: sample-range and window accounting; stop racing a recorder
  callback; queued work with no new input; slow inference; late worker
  completion; cancellation.

## Acceptance

- In TNT: `uv run ruff check src/ tests/` and
  `uv run python -m pytest tests/ -q`. In mlx-speech: `pytest tests/unit/`
  for the library prerequisites.
- A real live TNT session shows intermediate commits while a backlog drains
  and finalizes the full captured range. Fake-caller tests alone are not
  acceptance.
