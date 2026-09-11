# Retire the project website publisher

### Slice 1: Retire duplicate website ownership

**Objective:** Transfer website responsibility without changing library behavior.

**Acceptance criteria:** Preserve the original website URL and file contents;
remove the duplicate publisher only after central delivery is verified.

**Verification:** Run the library unit suite, the frontend model checks, and
live publication provenance checks. Compare runtime and documentation trees.

1. Preserve the deployed site in the frontend module and verify central delivery.
2. Remove `site/` and `.github/workflows/pages.yml`; document the new source.
3. Move HTML catalog assertions to the frontend test suite and retain README and
   model-guide checks in this repository.
4. Run the full unit suite and verify that runtime, guides, and examples are unchanged.
5. Merge only after the central publisher marker and original URLs are verified.

Status: complete. Central delivery is verified; the cleanup is ready to merge.

## Verification

The full library unit suite passes: 1,117 tests. Five website assertions moved
to the frontend suite; the original combined README/site test became a library
README and guide test. All 27 frontend tests pass, including the transferred
model checks. Runtime, technical guides, and examples have no changes.

The local Xcode compiler requires its matching macOS SDK for CPU compilation.
The test invocation set SDKROOT to Xcode's MacOSX.sdk for this process only;
no system configuration or model code was changed.


The original public route and canonical return successfully. All 18 non-audio
website files match the imported source; the recording matches within the
published artifact. The ordinary publisher marker identifies the frontend
repository at fea3d92e. The old Pages configuration is retired.
