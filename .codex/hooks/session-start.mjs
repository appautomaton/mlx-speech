import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildSessionContext } from '../../.agent/.automaton/lib/context.mjs'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..')

// Claude Code writes the hook payload as JSON on stdin, with `source` set to
// 'compact' after compaction. Hosts that pass no payload (or a non-JSON one)
// leave `compacted` false, so the shared hook stays safe on every host.
let payload = ''
if (!process.stdin.isTTY) {
  for await (const chunk of process.stdin) payload += chunk
}
let compacted = false
try {
  compacted = JSON.parse(payload).source === 'compact'
} catch {
  compacted = false
}

process.stdout.write(JSON.stringify({
  hookSpecificOutput: {
    hookEventName: 'SessionStart',
    additionalContext: buildSessionContext(projectRoot, { compacted })
  }
}) + '\n')
