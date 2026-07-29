// Pristine steering bodies. These live in the installed runtime rather than the
// CLI because two consumers need them and they must never drift apart: the
// scaffold writes them on install, and uninstall matches against them.
//
// One file only. Automaton owns the running log of work, not a description of
// the project: `ROADMAP.md` is the forward queue and cannot be true without the
// harness, while identity and constraints are already true in the repo's own
// README, AGENTS.md, and docs/. The retired PROJECT.md and REQUIREMENTS.md
// bodies moved to DEPRECATED_STEERING_FILES in lib/scaffold.mjs (DD-016).
export const STEERING_FILES = {
  'ROADMAP.md': '# Roadmap\n\nNo active roadmap.\n\n## Deferred or Not Now\n\n- None recorded.\n'
}
