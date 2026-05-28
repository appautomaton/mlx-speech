#!/usr/bin/env node
/**
 * sync-status.mjs
 *
 * Updates current.json through shared contract checks when state flags are provided.
 *
 * Usage:
 *   node sync-status.mjs [root=.]
 *   node sync-status.mjs [root=.] --active-change <slug> --stage <stage>
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

function loadContracts() {
  const candidates = [
    resolve(__dirname, '..', 'lib', 'contracts-data.json'),
    resolve(__dirname, '..', '..', '..', 'lib', 'contracts-data.json'),
    resolve(__dirname, '..', '..', '..', 'runtime', 'lib', 'contracts-data.json'),
    join(process.cwd(), '.agent', '.automaton', 'lib', 'contracts-data.json')
  ]
  for (const p of candidates) {
    if (existsSync(p)) {
      try { return JSON.parse(readFileSync(p, 'utf8')) } catch { /* next */ }
    }
  }
  return null
}

const contracts = loadContracts()

const STATE_FIELDS = [
  { camel: 'activeChange', snake: 'active_change', flag: '--active-change' },
  { camel: 'stage', snake: 'stage', flag: '--stage' },
  { camel: 'canonicalSpec', snake: 'canonical_spec', flag: '--canonical-spec' },
  { camel: 'canonicalDesign', snake: 'canonical_design', flag: '--canonical-design' },
  { camel: 'canonicalPlan', snake: 'canonical_plan', flag: '--canonical-plan' },
  { camel: 'productReview', snake: 'product_review', flag: '--product-review' },
  { camel: 'engineeringReview', snake: 'engineering_review', flag: '--engineering-review' }
]

const FLAG_TO_FIELD = new Map(STATE_FIELDS.map((field) => [field.flag, field]))
const KNOWN_STATE_KEYS = new Set(STATE_FIELDS.flatMap(({ camel, snake }) => [camel, snake]))

function diagnostic(level, code, message) {
  return { level, code, message }
}

function unique(values) {
  return [...new Set(values)]
}

function parseArgs(argv) {
  const args = {
    root: '.',
    rootSet: false,
    patch: {},
    changed: [],
    diagnostics: []
  }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--root') {
      const value = argv[i + 1]
      if (value === undefined || value.startsWith('--')) {
        args.diagnostics.push(diagnostic('error', 'missing_argument', '--root requires a value'))
        continue
      }
      args.root = value
      args.rootSet = true
      i += 1
      continue
    }

    const stateField = FLAG_TO_FIELD.get(arg)
    if (stateField) {
      const value = argv[i + 1]
      if (value === undefined || value.startsWith('--')) {
        args.diagnostics.push(diagnostic('error', 'missing_argument', `${arg} requires a value`))
        continue
      }
      args.patch[stateField.camel] = value
      args.changed.push(stateField.snake)
      i += 1
      continue
    }

    if (arg.startsWith('--')) {
      args.diagnostics.push(diagnostic('error', 'unknown_argument', `unknown argument: ${arg}`))
      continue
    }

    if (!args.rootSet) {
      args.root = arg
      args.rootSet = true
      continue
    }

    args.diagnostics.push(diagnostic('error', 'unexpected_argument', `unexpected positional argument: ${arg}`))
  }

  return {
    root: args.root,
    patch: args.patch,
    changed: unique(args.changed),
    diagnostics: args.diagnostics
  }
}

function normalizeCurrentState(parsed) {
  const source = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  const normalized = {}

  for (const { camel, snake } of STATE_FIELDS) {
    if (source[camel] !== undefined || source[snake] !== undefined) {
      normalized[camel] = source[camel] ?? source[snake]
    }
  }

  for (const [key, value] of Object.entries(source)) {
    if (!KNOWN_STATE_KEYS.has(key)) {
      normalized[key] = value
    }
  }

  return normalized
}

function serializeCurrentState(state) {
  const normalized = normalizeCurrentState(state)
  const out = {}

  for (const { camel, snake } of STATE_FIELDS) {
    if (normalized[camel] !== undefined && normalized[camel] !== null) {
      out[snake] = normalized[camel]
    }
  }

  for (const [key, value] of Object.entries(normalized)) {
    if (!KNOWN_STATE_KEYS.has(key)) {
      out[key] = value
    }
  }

  return out
}

function diagnoseState(state, projectRoot) {
  if (contracts === null) {
    return [diagnostic('warning', 'contracts_manifest_missing', 'contract manifest not found; semantic diagnostics skipped')]
  }

  const diagnostics = []
  const validStages = new Set(contracts.stages ?? [])
  const productVerdicts = new Set(contracts.reviewVerdicts?.product ?? [])
  const engineeringVerdicts = new Set(contracts.reviewVerdicts?.engineering ?? [])
  let invalidStage = false
  let stageIsValid = false

  if (state.activeChange === undefined || state.activeChange === 'none') {
    diagnostics.push(diagnostic('error', 'missing_active_change', 'no active change recorded'))
  }

  if (state.stage === undefined || state.stage === 'none') {
    diagnostics.push(diagnostic('error', 'missing_stage', 'no stage recorded'))
  } else if (!validStages.has(state.stage)) {
    diagnostics.push(diagnostic('error', 'invalid_stage', `invalid stage: ${state.stage}`))
    invalidStage = true
  } else {
    stageIsValid = true
  }

  if (stageIsValid) {
    for (const field of contracts.stagePrerequisites?.[state.stage] ?? []) {
      if (state[field] === undefined || state[field] === null) {
        const code = contracts.prerequisiteDiagnosticCodes?.[field] ?? `missing_${field}`
        diagnostics.push(diagnostic('error', code, `${state.stage} stage requires ${field}`))
      }
    }

    if (state.productReview !== undefined && state.productReview !== null) {
      if (!productVerdicts.has(state.productReview)) {
        diagnostics.push(diagnostic('error', 'invalid_product_review', `unrecognized product_review verdict: ${state.productReview}`))
      }
    }

    if (state.engineeringReview !== undefined && state.engineeringReview !== null) {
      if (!engineeringVerdicts.has(state.engineeringReview)) {
        diagnostics.push(diagnostic('error', 'invalid_engineering_review', `unrecognized engineering_review verdict: ${state.engineeringReview}`))
      }
    }
  }

  if (!invalidStage) {
    for (const { field, code, level } of contracts.canonicalPointerChecks ?? []) {
      if (state[field] && !existsSync(join(projectRoot, state[field]))) {
        diagnostics.push(diagnostic(level, code, `${field} points to ${state[field]} but file does not exist`))
      }
    }
  }

  return diagnostics
}

function loadExistingState(statePath) {
  if (!existsSync(statePath)) {
    return { state: {}, exists: false, diagnostics: [] }
  }

  try {
    return {
      state: normalizeCurrentState(JSON.parse(readFileSync(statePath, 'utf8'))),
      exists: true,
      diagnostics: []
    }
  } catch (err) {
    return {
      state: {},
      exists: true,
      diagnostics: [diagnostic('error', 'invalid_state_json', `current.json is not valid JSON: ${err.message}`)]
    }
  }
}

function writeCurrentState(statePath, state) {
  mkdirSync(dirname(statePath), { recursive: true })
  writeFileSync(statePath, JSON.stringify(serializeCurrentState(state), null, 2) + '\n', 'utf8')
}

function clearUnlessPatched(state, patch, fields) {
  for (const field of fields) {
    if (patch[field] === undefined) {
      delete state[field]
    }
  }
}

function applyStatePatch(currentState, patch) {
  const nextState = {
    ...currentState,
    ...patch
  }

  if (patch.activeChange !== undefined && patch.activeChange !== currentState.activeChange) {
    clearUnlessPatched(nextState, patch, [
      'canonicalSpec',
      'canonicalDesign',
      'canonicalPlan',
      'productReview',
      'engineeringReview'
    ])
  }

  if (patch.canonicalSpec !== undefined && patch.canonicalSpec !== currentState.canonicalSpec) {
    clearUnlessPatched(nextState, patch, [
      'canonicalDesign',
      'canonicalPlan',
      'productReview',
      'engineeringReview'
    ])
  }

  if (patch.canonicalPlan !== undefined && patch.canonicalPlan !== currentState.canonicalPlan) {
    clearUnlessPatched(nextState, patch, ['engineeringReview'])
  }

  return nextState
}

const args = parseArgs(process.argv.slice(2))
const root = args.root
const statePath = join(root, '.agent', '.automaton', 'state', 'current.json')

if (args.diagnostics.some((item) => item.level === 'error')) {
  console.log(JSON.stringify({
    synced: false,
    statePath,
    diagnostics: args.diagnostics
  }, null, 2))
  process.exit(1)
}

const result = {
  synced: true,
  statePath
}

if (args.changed.length > 0) {
  const loaded = loadExistingState(statePath)
  const nextState = applyStatePatch(loaded.state, args.patch)
  const diagnostics = loaded.diagnostics.length > 0
    ? loaded.diagnostics
    : diagnoseState(nextState, resolve(root))
  const hasError = diagnostics.some((item) => item.level === 'error')

  result.statePath = statePath
  result.stateChanged = false
  result.changed = args.changed
  result.diagnostics = diagnostics

  if (hasError) {
    result.synced = false
    console.log(JSON.stringify(result, null, 2))
    process.exit(1)
  }

  const before = loaded.exists ? JSON.stringify(serializeCurrentState(loaded.state)) : null
  const after = JSON.stringify(serializeCurrentState(nextState))
  result.stateChanged = before !== after

  if (result.stateChanged) {
    writeCurrentState(statePath, nextState)
  }
}

console.log(JSON.stringify(result, null, 2))
