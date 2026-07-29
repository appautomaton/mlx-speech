import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { loadCurrentState } from './state.mjs'
import { validateHandoff } from './validate.mjs'

const STATE_PATH = '.agent/.automaton/state/current.json'
const RECEIPT_PATH = '.agent/.automaton/state/install-manifest.json'

// The session hook is the only Automaton surface a user sees without asking for
// it, so it carries facts the harness knows and the user cannot: state that does
// not resolve, work that finished without advancing, an install that cannot
// prune itself. Checks needing the CLI source tree (version skew, orphaned skill
// directories) stay with `status`, which has the comparison target this runtime
// does not.
//
// Steering is deliberately unchecked. `ROADMAP.md` is the only steering file,
// and "No active roadmap" is a legitimate steady state, so flagging it would be
// noise forever (DD-016).

function readIfPresent(target) {
  if (!existsSync(target)) {
    return null
  }

  try {
    return readFileSync(target, 'utf8')
  } catch {
    return null
  }
}

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

// A plan whose slices all report status, at a stage that never advanced, is
// finished work nobody was told about. This is the failure that leaves a change
// parked for weeks while unrelated commits pile up on top of it.
function parkedChange(state, projectRoot) {
  if (state.stage !== 'execute' && state.stage !== 'verify') {
    return null
  }

  if (!state.canonicalPlan) {
    return null
  }

  const plan = readIfPresent(join(projectRoot, state.canonicalPlan))
  if (plan === null) {
    return null
  }

  const slices = [...plan.matchAll(/^### Slice\b/gm)].length
  const reported = [...plan.matchAll(/^\*\*Status:\*\*/gm)].length

  if (slices === 0 || reported < slices) {
    return null
  }

  if (/^\*\*Status:\*\*\s*blocked/m.test(plan)) {
    return null
  }

  return `all ${slices} slices in ${state.canonicalPlan} report status and none are blocked, but the stage is still ${state.stage}. The change may have finished or parked at a checkpoint without advancing: read the plan before starting unrelated work.`
}

function missingReceipt(projectRoot) {
  if (existsSync(join(projectRoot, RECEIPT_PATH))) {
    return null
  }

  return 'no install receipt: this install predates receipt tracking, so an upgrade cannot prune skills the source has removed. Reinstall to write one.'
}

export function sessionHealthFindings(projectRoot, state) {
  const findings = []

  if (state) {
    for (const item of validateHandoff(state, projectRoot).diagnostics) {
      findings.push(`${item.level}: ${item.message}`)
    }

    const parked = parkedChange(state, projectRoot)
    if (parked !== null) {
      findings.push(parked)
    }
  }

  const receipt = missingReceipt(projectRoot)
  if (receipt !== null) {
    findings.push(receipt)
  }

  return findings
}

export function buildSessionContext(projectRoot, options = {}) {
  const { compacted = false } = options
  const state = loadCurrentStateSummary(projectRoot)
  const messages = []

  messages.push('<automaton_reminder>')
  messages.push('Automaton is installed for this project as a stage-gated workflow.')

  if (state?.disengaged) {
    messages.push(`Change "${state.activeChange}" is verified and the harness is disengaged until your next objective.`)
  } else {
    if (state?.activeChange && state?.stage) {
      messages.push(`Current state: ${STATE_PATH} (change=${state.activeChange}; stage=${state.stage}).`)
    } else {
      messages.push(`Current state: ${STATE_PATH} (no active change recorded).`)
    }

    messages.push('Read .agent/.automaton/references/FRAMEWORK.md once per session to refresh the operating model.')
  }

  messages.push("Treat this as orientation, not a mandate. The user's latest request stays in charge.")

  const findings = sessionHealthFindings(projectRoot, state)
  if (findings.length > 0) {
    messages.push('Needs attention before you trust this state:')
    for (const finding of findings) {
      messages.push(`- ${finding}`)
    }
  }

  if (compacted) {
    messages.push('This session was compacted. Reload current.json and the relevant work artifacts before relying on prior Automaton context.')
  }

  messages.push('</automaton_reminder>')

  return messages.join('\n')
}
