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
import { spawnSync } from 'node:child_process'
import { readFileSync, existsSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_STATE = {
  activeChange: 'none',
  stage: 'none',
  canonicalSpec: null,
  canonicalDesign: null,
  canonicalPlan: null,
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
    engineering_review: engineeringReviewSnake,
    activeChange,
    canonicalSpec,
    canonicalDesign,
    canonicalPlan,
    engineeringReview,
    ...rest
  } = state

  return {
    activeChange: activeChange ?? activeChangeSnake ?? DEFAULT_STATE.activeChange,
    stage: state.stage ?? DEFAULT_STATE.stage,
    canonicalSpec: canonicalSpec ?? canonicalSpecSnake ?? null,
    canonicalDesign: canonicalDesign ?? canonicalDesignSnake ?? null,
    canonicalPlan: canonicalPlan ?? canonicalPlanSnake ?? null,
    engineeringReview: engineeringReview ?? engineeringReviewSnake ?? null,
    ...Object.fromEntries(
      Object.entries(rest).filter(
        ([key]) => ![
          'active_change',
          'stage',
          'canonical_spec',
          'canonical_design',
          'canonical_plan',
          'engineering_review'
        ].includes(key)
      )
    )
  }
}

function diagnose(state, projectRoot, manifest) {
  const diagnostics = []
  const validStages = new Set(manifest.stages ?? [])
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

// L2 artifact-shape lint. Warning-level only: it surfaces gaps for the reading
// skill to judge, it never blocks. L1 stage/pointer checks above stay the only
// error-level gate.
function lintArtifacts(state, projectRoot, manifest) {
  const lint = manifest.artifactLint
  if (!lint || projectRoot === null) {
    return []
  }

  const diagnostics = []

  if (state.canonicalSpec) {
    const specPath = join(projectRoot, state.canonicalSpec)
    if (existsSync(specPath)) {
      const spec = readFileSync(specPath, 'utf8')
      for (const check of lint.spec ?? []) {
        if (!new RegExp(check.pattern, 'i').test(spec)) {
          diagnostics.push(diagnostic('warning', check.code, check.message))
        }
      }
    }
  }

  // A review verdict in state with no matching section on the canonical
  // artifact means the two surfaces drifted: the verdict was synced but the
  // rationale never landed, or the artifact was rewritten over it.
  for (const check of lint.reviewSections ?? []) {
    if (!state[check.verdictField] || !state[check.artifactField]) {
      continue
    }
    const artifactPath = join(projectRoot, state[check.artifactField])
    if (existsSync(artifactPath) && !readFileSync(artifactPath, 'utf8').includes(check.heading)) {
      diagnostics.push(diagnostic('warning', check.code, check.message))
    }
  }

  if (state.canonicalPlan && lint.planSliceHeading) {
    const planPath = join(projectRoot, state.canonicalPlan)
    if (existsSync(planPath)) {
      const plan = readFileSync(planPath, 'utf8')
      const headings = [...plan.matchAll(new RegExp(lint.planSliceHeading, 'gm'))]

      if (headings.length === 0 && lint.planMissingSlices) {
        diagnostics.push(diagnostic('warning', lint.planMissingSlices.code, lint.planMissingSlices.message))
      }

      for (let i = 0; i < headings.length; i += 1) {
        const end = i + 1 < headings.length ? headings[i + 1].index : plan.length
        const body = plan.slice(headings[i].index, end)
        for (const field of lint.planSliceFields ?? []) {
          if (!body.includes(field.label)) {
            const fieldName = field.label.replaceAll('*', '').replace(':', '')
            diagnostics.push(diagnostic('warning', field.code, `slice ${headings[i][1]} lacks ${fieldName}`))
          }
        }
      }
    }
  }

  return diagnostics
}

// L2 drift hints. The state cursor can silently lag the repository when work
// happens outside the harness. These warnings surface the lag for the reading
// skill to judge; they never block, and they degrade silently to nothing in
// projects without git.
function gitLines(projectRoot, args) {
  const result = spawnSync('git', args, { cwd: projectRoot, encoding: 'utf8' })
  if (result.status !== 0 || typeof result.stdout !== 'string') {
    return null
  }
  return result.stdout.split('\n').filter(Boolean)
}

function driftDiagnostics(target, state, projectRoot) {
  if (projectRoot === null) {
    return []
  }

  const diagnostics = []
  const stateMtime = statSync(target).mtime.toISOString()
  const commitsSince = gitLines(projectRoot, ['log', '--oneline', `--since=${stateMtime}`, '-20'])
  // Per-slice rhythm commits are the harness's own execution ledger: state syncs
  // once at execute entry, then slices commit without touching current.json, so
  // only out-of-band commits count as drift.
  const outOfBand = commitsSince === null
    ? null
    : commitsSince.filter((line) => !/^\S+ slice \S+(?: gap-fix)?: /.test(line))

  if (outOfBand !== null && outOfBand.length >= 3) {
    diagnostics.push(diagnostic(
      'warning',
      'state_drift',
      `${outOfBand.length}${commitsSince.length === 20 ? '+' : ''} out-of-band commits since current.json last changed; harness state may lag the repo. Refresh state through the lifecycle.`
    ))
  }

  if (state.stage === 'verified') {
    const dirty = gitLines(projectRoot, ['status', '--porcelain'])
    if (dirty !== null && dirty.length >= 5) {
      diagnostics.push(diagnostic(
        'warning',
        'dirty_tree_at_verified',
        `stage is verified but the working tree has ${dirty.length} changed paths; work may be happening outside a tracked change`
      ))
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

  const projectRoot = deriveProjectRoot(target)
  return [
    ...diagnose(state, projectRoot, manifest),
    ...lintArtifacts(state, projectRoot, manifest),
    ...driftDiagnostics(target, state, projectRoot)
  ]
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
