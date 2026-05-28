import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { isValidStage } from './contracts.mjs'

const FIELD_MAP = [
  ['activeChange', 'active_change'],
  ['canonicalSpec', 'canonical_spec'],
  ['canonicalDesign', 'canonical_design'],
  ['canonicalPlan', 'canonical_plan'],
  ['productReview', 'product_review'],
  ['engineeringReview', 'engineering_review']
]

const ALL_KNOWN_KEYS = new Set(FIELD_MAP.flatMap(([camel, snake]) => [camel, snake]))

export function normalizeCurrentState(state) {
  const normalized = {}

  for (const [camel, snake] of FIELD_MAP) {
    if (state[camel] !== undefined || state[snake] !== undefined) {
      normalized[camel] = state[camel] ?? state[snake]
    }
  }

  for (const [key, val] of Object.entries(state)) {
    if (!ALL_KNOWN_KEYS.has(key)) {
      normalized[key] = val
    }
  }

  return normalized
}

function validateCurrentState(state) {
  if (state.activeChange === undefined) {
    throw new Error('invalid current state: missing active change')
  }

  if (state.stage === undefined) {
    throw new Error('invalid current state: missing stage')
  }

  if (!isValidStage(state.stage)) {
    throw new Error(`invalid stage: ${state.stage}`)
  }

  return state
}

function serializeCurrentState(state) {
  const normalized = validateCurrentState(normalizeCurrentState(state))
  const out = {}

  if (normalized.activeChange !== undefined) out.active_change = normalized.activeChange
  if (normalized.stage !== undefined) out.stage = normalized.stage

  for (const [camel, snake] of FIELD_MAP) {
    if (camel === 'activeChange') continue
    if (normalized[camel] !== undefined) out[snake] = normalized[camel]
  }

  for (const [key, val] of Object.entries(normalized)) {
    if (key === 'stage' || ALL_KNOWN_KEYS.has(key)) continue
    out[key] = val
  }

  return out
}

export function saveCurrentState(target, state) {
  const normalized = validateCurrentState(normalizeCurrentState(state))

  mkdirSync(dirname(target), { recursive: true })
  writeFileSync(target, JSON.stringify(serializeCurrentState(normalized), null, 2) + '\n', 'utf8')
}

export function loadCurrentState(target) {
  return validateCurrentState(normalizeCurrentState(JSON.parse(readFileSync(target, 'utf8'))))
}
