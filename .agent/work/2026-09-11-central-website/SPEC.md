# Central website publication

Move responsibility for the public website to the App Automaton frontend
repository while preserving the public address, content, media, and design.
Remove the obsolete site copy and Pages workflow only after central delivery is
verified. Library code, model guides, examples, and model weights remain here.
Keep README and guide assertions in library tests; move website assertions with
the website. Never play media during validation.

## Acceptance criteria

The original website URL is served by the frontend publisher, with unchanged
page and asset bytes. This repository contains no website publisher or duplicate
site tree. All library unit checks pass, and frontend model checks cover the
transferred catalog assertions.

## Anti-goals

No runtime, API, model, checkpoint, example, or technical-guide changes.
