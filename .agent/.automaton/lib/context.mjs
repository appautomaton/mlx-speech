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
  messages.push('This project has the Automaton stage-gated harness installed.')

  if (activeChange && stage) {
    messages.push(`State JSON: ${STATE_PATH} (change=${activeChange}; stage=${stage}).`)
  } else {
    messages.push(`State JSON: ${STATE_PATH} (no active state recorded).`)
  }

  messages.push('Work artifacts, when relevant, live under .agent/work/; canonical artifact pointers are in current.json.')
  messages.push('Read .agent/.automaton/references/FRAMEWORK.md once per session for the operating model.')
  messages.push("Reminder only: honor the user's latest request; use Automaton files when relevant, not as a mandate.")
  messages.push('Vocabulary: change, stage, slice, artifact, steering.')

  if (compacted) {
    messages.push('Context compacted; reload current.json and relevant work artifacts before relying on prior Automaton context.')
  }

  messages.push('</automaton_reminder>')

  return messages.join(' ')
}
