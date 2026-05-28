import { readFileSync } from 'node:fs'

export const CONTRACTS_DATA = JSON.parse(
  readFileSync(new URL('./contracts-data.json', import.meta.url), 'utf8')
)

export const STAGES = CONTRACTS_DATA.stages

export const LENSES = CONTRACTS_DATA.lenses

export const ARTIFACT_LAYOUT = CONTRACTS_DATA.artifactLayout

export function isValidStage(stage) {
  return STAGES.includes(stage)
}

export function isValidLens(lens) {
  return LENSES.includes(lens)
}

export const EXECUTION_ROUTES = CONTRACTS_DATA.executionRoutes

export const CHECKPOINT_TYPES = CONTRACTS_DATA.checkpointTypes

export const TOPOLOGY_LABELS = CONTRACTS_DATA.topologyLabels

export function isValidExecutionRoute(route) {
  return EXECUTION_ROUTES.includes(route)
}

export function isValidCheckpointType(checkpoint) {
  return CHECKPOINT_TYPES.includes(checkpoint)
}

export const STAGE_PREREQUISITES = CONTRACTS_DATA.stagePrerequisites

export const PRODUCT_REVIEW_VERDICTS = CONTRACTS_DATA.reviewVerdicts.product

export const ENGINEERING_REVIEW_VERDICTS = CONTRACTS_DATA.reviewVerdicts.engineering

export const PREREQUISITE_DIAGNOSTIC_CODES = CONTRACTS_DATA.prerequisiteDiagnosticCodes

export const CANONICAL_POINTER_CHECKS = CONTRACTS_DATA.canonicalPointerChecks

export function isValidProductReview(verdict) {
  return PRODUCT_REVIEW_VERDICTS.includes(verdict)
}

export function isValidEngineeringReview(verdict) {
  return ENGINEERING_REVIEW_VERDICTS.includes(verdict)
}
