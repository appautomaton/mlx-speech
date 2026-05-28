# Debug Protocol

Extended guidance for systematic debugging.

## Common Root Cause Patterns

### Node.js

- **"Cannot find module"**: Check `node_modules` exists. Check `package.json` has the dependency. Check import path spelling. Check if the package is ESM-only and you're using `require()`.
- **Async test timeout**: Check for unhandled promise rejections. Check for missing `await`. Check if the test is actually calling `done()` or returning a promise.
- **Memory leak in tests**: Check for event listeners not being removed. Check for global mocks not being restored.

### Python

- **ImportError**: Check virtual environment is active. Check `PYTHONPATH`. Check for circular imports.
- **AssertionError in tests**: Check if the test is comparing floats with `==`. Check if the test depends on dictionary ordering (pre-3.7). Check timezone handling.
- **Database locked (SQLite)**: Check for unclosed connections. Check for transactions left open.

### Rust

- **Borrow checker errors**: Check ownership semantics. Check if `Rc<RefCell<T>>` or `Arc<Mutex<T>>` is needed. Check lifetime annotations.
- **Panic in tests**: Check for `unwrap()` on `None` or `Err`. Check for out-of-bounds indexing.

### General

- **"It works on my machine"**: Check environment variables. Check file paths (case sensitivity on macOS/Linux vs. Windows). Check line endings.
- **Heisenbug**: Check for race conditions. Check for uninitialized memory. Check for dependency on system time or randomness.

## Investigation Techniques

1. **Bisection.** If a test suite has 100 tests and 1 fails, run the first 50. If they pass, the failure is in the second 50. Repeat until isolated.
2. **Minimal reproduction.** Remove code until the bug disappears. The last thing removed is the trigger.
3. **Contrast.** Find a similar test or function that works. Compare line by line until you find the difference.
4. **Logging.** Add `console.log`, `print`, or `eprintln` at key points. Do not use a debugger unless the logs are insufficient; debuggers are slower and more disruptive.

## Escalation Template

If you cannot isolate the root cause within 3 attempts, report to the user with:

```
**Observed:** [what the system does]
**Expected:** [what the system should do]
**Tried:** [what you investigated]
**Need:** [what you need from the user to proceed]
```

Example:
```
**Observed:** `npm test` fails with "Cannot find module '../config'" in 3 files.
**Expected:** Tests should resolve the config module.
**Tried:** Verified `src/config/index.js` exists. Verified `package.json` main field. Checked for typos in import paths.
**Need:** Is there a build step or alias configuration that resolves this path? I don't see it in the plan.
```
