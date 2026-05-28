#!/usr/bin/env node
/**
 * get-context.mjs
 *
 * Reads the Automaton current state and outputs normalized camelCase JSON.
 * If current.json does not exist, returns the same deterministic key shape with
 * activeChange/stage set to "none" and canonical/review pointers set to null.
 *
 * Usage: node get-context.mjs [path/to/current.json]
 */
import { readFileSync, existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_STATE = {
  activeChange: 'none',
  stage: 'none',
  canonicalSpec: null,
  canonicalDesign: null,
  canonicalPlan: null,
  productReview: null,
  engineeringReview: null
}

function diagnostic(level, code, message) {
  return { level, code, message }
}

function deriveProjectRoot(currentJsonPath) {
  const resolved = resolve(currentJsonPath)
  const candidate = dirname(dirname(dirname(dirname(resolved))))
  if (resolve(join(candidate, '.agent', '.automaton', 'state', 'current.json')) !== resolved) {
    return null
  }
  return candidate
}

function unique(values) {
  return [...new Set(values.filter(Boolean))]
}

// Keep this script self-contained because skills invoke it from installed
// project runtime state where package source imports may not resolve.
function contractManifestCandidates(currentJsonPath) {
  const scriptDir = dirname(fileURLToPath(import.meta.url))
  const projectRoot = deriveProjectRoot(currentJsonPath)

  return unique([
    projectRoot === null ? null : join(projectRoot, '.agent', '.automaton', 'lib', 'contracts-data.json'),
    join(process.cwd(), '.agent', '.automaton', 'lib', 'contracts-data.json'),
    resolve(scriptDir, '..', 'lib', 'contracts-data.json'),
    resolve(scriptDir, '..', '..', '..', 'runtime', 'lib', 'contracts-data.json')
  ])
}

function loadContractManifest(currentJsonPath) {
  for (const candidate of contractManifestCandidates(currentJsonPath)) {
    if (!existsSync(candidate)) {
      continue
    }

    try {
      return JSON.parse(readFileSync(candidate, 'utf8'))
    } catch {
      // Try the next candidate; installed shared scripts should stay usable.
    }
  }

  return null
}

function normalizeCurrentState(parsed) {
  const state = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  const {
    active_change: activeChangeSnake,
    canonical_spec: canonicalSpecSnake,
    canonical_design: canonicalDesignSnake,
    canonical_plan: canonicalPlanSnake,
    product_review: productReviewSnake,
    engineering_review: engineeringReviewSnake,
    activeChange,
    canonicalSpec,
    canonicalDesign,
    canonicalPlan,
    productReview,
    engineeringReview,
    ...rest
  } = state

  return {
    activeChange: activeChange ?? activeChangeSnake ?? DEFAULT_STATE.activeChange,
    stage: state.stage ?? DEFAULT_STATE.stage,
    canonicalSpec: canonicalSpec ?? canonicalSpecSnake ?? null,
    canonicalDesign: canonicalDesign ?? canonicalDesignSnake ?? null,
    canonicalPlan: canonicalPlan ?? canonicalPlanSnake ?? null,
    productReview: productReview ?? productReviewSnake ?? null,
    engineeringReview: engineeringReview ?? engineeringReviewSnake ?? null,
    ...Object.fromEntries(
      Object.entries(rest).filter(
        ([key]) => ![
          'active_change',
          'stage',
          'canonical_spec',
          'canonical_design',
          'canonical_plan',
          'product_review',
          'engineering_review'
        ].includes(key)
      )
    )
  }
}

function diagnose(state, projectRoot, manifest) {
  const diagnostics = []
  const validStages = new Set(manifest.stages ?? [])
  const productVerdicts = new Set(manifest.reviewVerdicts?.product ?? [])
  const engineeringVerdicts = new Set(manifest.reviewVerdicts?.engineering ?? [])
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
    for (const field of manifest.stagePrerequisites?.[state.stage] ?? []) {
      if (state[field] === undefined || state[field] === null) {
        const code = manifest.prerequisiteDiagnosticCodes?.[field] ?? `missing_${field}`
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

  if (!invalidStage && projectRoot !== null) {
    for (const { field, code, level } of manifest.canonicalPointerChecks ?? []) {
      if (state[field] && !existsSync(join(projectRoot, state[field]))) {
        diagnostics.push(diagnostic(level, code, `${field} points to ${state[field]} but file does not exist`))
      }
    }
  }

  return diagnostics
}

function diagnosticsFor(target, state, stateExists) {
  const manifest = loadContractManifest(target)
  if (manifest === null) {
    return [diagnostic('warning', 'contracts_manifest_missing', 'contract manifest not found; semantic diagnostics skipped')]
  }

  if (!stateExists) {
    return []
  }

  return diagnose(state, deriveProjectRoot(target), manifest)
}

const target = process.argv[2] ?? join('.agent', '.automaton', 'state', 'current.json')

if (!existsSync(target)) {
  const diagnostics = diagnosticsFor(target, DEFAULT_STATE, false)
  console.log(JSON.stringify({ ...DEFAULT_STATE, diagnostics }, null, 2))
  process.exit(0)
}

try {
  const raw = readFileSync(target, 'utf8')
  const parsed = JSON.parse(raw)
  const normalized = normalizeCurrentState(parsed)
  const diagnostics = diagnosticsFor(target, normalized, true)

  console.log(JSON.stringify({ ...normalized, diagnostics }, null, 2))
} catch (err) {
  console.log(JSON.stringify({
    ...DEFAULT_STATE,
    diagnostics: [diagnostic('error', 'invalid_state_json', `current.json is not valid JSON: ${err.message}`)]
  }, null, 2))
  process.exit(1)
}
