import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildSessionContext } from '../../.agent/.automaton/lib/context.mjs'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..')

process.stdout.write(JSON.stringify({
  hookSpecificOutput: {
    hookEventName: 'SessionStart',
    additionalContext: buildSessionContext(projectRoot)
  }
}) + '\n')
