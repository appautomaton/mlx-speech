import { existsSync } from 'node:fs'
import { join } from 'node:path'

import { loadCurrentState } from './state.mjs'

const STATE_PATH = '.agent/.automaton/state/current.json'

function loadCurrentStateSummary(projectRoot) {
  const target = join(projectRoot, STATE_PATH)

  if (!existsSync(target)) {
    return null
  }

  try {
    return loadCurrentState(target)
  } catch {
    return null
  }
}

export function buildSessionContext(projectRoot, options = {}) {
  const { compacted = false } = options
  const state = loadCurrentStateSummary(projectRoot)
  const activeChange = state?.activeChange
  const stage = state?.stage
  const messages = []

  messages.push('<automaton_reminder>')
  messages.push('Automaton is installed for this project as a stage-gated workflow.')

  if (activeChange && stage) {
    messages.push(`Current state: ${STATE_PATH} (change=${activeChange}; stage=${stage}).`)
  } else {
    messages.push(`Current state: ${STATE_PATH} (no active change recorded).`)
  }

  messages.push('Work artifacts live under .agent/work/ when they matter. Canonical artifact pointers live in current.json.')
  messages.push('Read .agent/.automaton/references/FRAMEWORK.md once per session to refresh the operating model.')
  messages.push("Treat this as orientation, not a mandate. The user's latest request stays in charge; use Automaton files only when they are relevant.")
  messages.push('Shared vocabulary: change, stage, slice, artifact, steering.')

  if (compacted) {
    messages.push('This session was compacted. Reload current.json and the relevant work artifacts before relying on prior Automaton context.')
  }

  messages.push('</automaton_reminder>')

  return messages.join('\n')
}
